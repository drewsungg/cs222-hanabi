from openai import OpenAI
from settings import OPENAI_API_KEY, LLM_VERS, DEBUG
import os
import random
from hanabi_learning_environment import rl_env
from hanabi_learning_environment.agents.simple_agent import SimpleAgent
from agents import agent_list
import pandas as pd
import numpy as np

# Create a single client instance to be shared
try:
    client = OpenAI(api_key=OPENAI_API_KEY)
except Exception as e:
    print(f"Warning: Could not initialize OpenAI client: {e}")
    client = None

class HanabiHybridAgent(SimpleAgent):
    def __init__(self, config, agent_data):
        super().__init__(config)
        self.name = agent_data["name"]
        self.persona = agent_data["persona"]
        self.all_player_names = []
        self.client = client

    def get_player_name(self, current_player, target_offset):
        """Get the name of a player based on current player and target offset."""
        if not self.all_player_names:
            return f"Player {target_offset}"
        target_player_id = (current_player + target_offset) % len(self.all_player_names)
        return self.all_player_names[target_player_id]

    def _format_known_cards(self, observation):
        """Format known card information for LLM."""
        known_cards = observation['card_knowledge'][0]
        formatted = []
        for i, card in enumerate(known_cards):
            info = []
            if card['color'] is not None:
                info.append(f"color: {card['color']}")
            if card['rank'] is not None:
                info.append(f"rank: {card['rank'] + 1}")
            if info:
                formatted.append(f"Card {i}: {', '.join(info)}")
            else:
                formatted.append(f"Card {i}: No information")
        return "\n".join(formatted)

    def _format_legal_moves(self, legal_moves, observation):
        """Format legal moves for LLM."""
        formatted = []
        for i, move in enumerate(legal_moves):
            if move['action_type'] == 'PLAY':
                card_info = self._get_card_info(observation, move['card_index'])
                formatted.append(f"{i}: Play card {move['card_index']} {card_info}")
            elif move['action_type'] == 'DISCARD':
                card_info = self._get_card_info(observation, move['card_index'])
                formatted.append(f"{i}: Discard card {move['card_index']} {card_info}")
            elif move['action_type'].startswith('REVEAL'):
                target_name = self.get_player_name(observation['current_player'], move['target_offset'])
                info_type = 'color' if move['action_type'] == 'REVEAL_COLOR' else 'rank'
                info_value = move['color'] if info_type == 'color' else move['rank'] + 1
                formatted.append(f"{i}: Tell {target_name} about {info_type} {info_value}")
        return "\n".join(formatted)

    def _get_card_info(self, observation, card_index):
        """Get known information about a card."""
        card_knowledge = observation['card_knowledge'][0][card_index]
        info = []
        if card_knowledge['color'] is not None:
            info.append(f"color: {card_knowledge['color']}")
        if card_knowledge['rank'] is not None:
            info.append(f"rank: {card_knowledge['rank'] + 1}")
        return f"({', '.join(info)})" if info else "(no information)"

    def get_player_weights(self, observation):
        """Calculate action weights based on persona traits and game state."""
        weights = {
            'PLAY': 0.33,
            'DISCARD': 0.33,
            'HINT': 0.33
        }
        
        persona_lower = str(self.persona).lower()
        
        # Analytical traits influence
        if any(trait in persona_lower for trait in ['analytical', 'logical', 'methodical']):
            if observation['information_tokens'] < 4:
                weights['DISCARD'] += 0.2
                weights['PLAY'] -= 0.1
                weights['HINT'] -= 0.1
        
        # Risk tolerance traits
        if any(trait in persona_lower for trait in ['confident', 'risk', 'adventurous']):
            if observation['life_tokens'] > 1:
                weights['PLAY'] += 0.2
                weights['DISCARD'] -= 0.1
                weights['HINT'] -= 0.1
                
        # Cautious traits
        if any(trait in persona_lower for trait in ['cautious', 'careful', 'conservative']):
            if observation['life_tokens'] == 1:
                weights['PLAY'] -= 0.2
                weights['HINT'] += 0.1
                weights['DISCARD'] += 0.1
        
        # Social traits influence
        if any(trait in persona_lower for trait in ['collaborative', 'social', 'helpful']):
            if observation['information_tokens'] > 0:
                weights['HINT'] += 0.2
                weights['PLAY'] -= 0.1
                weights['DISCARD'] -= 0.1

        # Achievement-oriented traits
        if any(trait in persona_lower for trait in ['achievement', 'ambitious', 'driven']):
            total_score = sum(observation['fireworks'].values())
            if total_score < 10:
                weights['PLAY'] += 0.15
                weights['DISCARD'] -= 0.15

        # MBTI-based adjustments
        if 'MBTI' in persona_lower:
            if any(type in persona_lower for type in ['intj', 'istj', 'intp', 'istp']):
                weights['HINT'] -= 0.1
                weights['DISCARD'] += 0.1
            elif any(type in persona_lower for type in ['enfj', 'esfj', 'enfp', 'esfp']):
                weights['HINT'] += 0.1
                weights['PLAY'] -= 0.1

        # Normalize weights
        for action in weights:
            weights[action] = max(0.1, min(0.9, weights[action]))
        total = sum(weights.values())
        return {k: v/total for k, v in weights.items()}

    def _get_fallback_analysis(self, observation):
        """Generate fallback analysis when LLM is unavailable."""
        life_tokens = observation['life_tokens']
        info_tokens = observation['information_tokens']
        fireworks = observation['fireworks']
        
        analysis = []
        if life_tokens == 1:
            analysis.append("Critical situation with one life remaining. Must play very safely.")
        elif life_tokens == 2:
            analysis.append("Need to be cautious with plays.")
        
        if info_tokens == 0:
            analysis.append("Must discard to regain information tokens.")
        elif info_tokens == 8:
            analysis.append("Should use information tokens for team coordination.")
        
        total_score = sum(fireworks.values())
        if total_score == 0:
            analysis.append("Need to establish first cards in stacks.")
        elif total_score > 20:
            analysis.append("Good progress, focus on completing stacks.")
        
        return " ".join(analysis)

    def _analyze_game_state(self, observation):
        """Generate strategic analysis using GPT."""
        if not self.client:
            return self._get_fallback_analysis(observation)
            
        prompt = f"""As {self.name}, a Hanabi player who {self.persona}, analyze this game state:
        
        Fireworks: {observation['fireworks']}
        Information tokens: {observation['information_tokens']}
        Life tokens: {observation['life_tokens']}
        
        My cards with known information:
        {self._format_known_cards(observation)}
        
        What's most important to consider? Think step by step about:
        1. Current game priorities
        2. Risk assessment
        3. Team needs
        4. Critical cards
        
        Provide strategic analysis in 3-4 sentences, focusing on concrete actions."""
        
        try:
            response = self.client.chat.completions.create(
                model=LLM_VERS,
                messages=[
                    {"role": "system", "content": "You are an expert Hanabi player giving strategic advice. Be concise and specific."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=150,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"API error in analysis: {e}")
            return self._get_fallback_analysis(observation)

    def _get_action_recommendation(self, observation, legal_moves):
        """Hybrid approach combining LLM reasoning with personality weights."""
        if not self.client:
            return self._get_fallback_move(observation, legal_moves), "Fallback move - LLM unavailable"

        # Get personality-based weights
        weights = self.get_player_weights(observation)
        
        # Get LLM recommendation
        prompt = f"""As {self.name}, considering your personality who {self.persona}, analyze these possible moves in Hanabi:
Game State:
- Fireworks: {observation['fireworks']}
- Info tokens: {observation['information_tokens']}
- Life tokens: {observation['life_tokens']}
Legal Moves:
{self._format_legal_moves(legal_moves, observation)}

Consider:
1. Playing cards with hints that are likely playable
2. Giving hints about playable cards
3. Playing cards with some hints if we have lives to spare
4. Discarding unhinted cards when low on information tokens

Which move would you choose and why?
Respond in this format:
MOVE: [index of chosen move]
REASONING: [1-2 sentences explaining why]"""

        try:
            response = self.client.chat.completions.create(
                model=LLM_VERS,
                messages=[
                    {"role": "system", "content": "You are a strategic Hanabi player making a specific move choice."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=150,
                temperature=0.7
            )
            
            content = response.choices[0].message.content
            move_line = [line for line in content.split('\n') if line.startswith('MOVE:')][0]
            reasoning_line = [line for line in content.split('\n') if line.startswith('REASONING:')][0]
            
            llm_move_index = int(move_line.split(':')[1].strip())
            reasoning = reasoning_line.split('REASONING:')[1].strip()
            
            # Apply personality weights to modify LLM decision
            if 0 <= llm_move_index < len(legal_moves):
                llm_move = legal_moves[llm_move_index]
                action_type = llm_move['action_type']
                if action_type.startswith('REVEAL'):
                    action_type = 'HINT'
                
                # If personality strongly suggests a different action type (1.5x higher weight)
                for move_type, weight in weights.items():
                    if weight > weights.get(action_type, 0) * 1.5:
                        # Look for alternative moves of the preferred type
                        alt_moves = []
                        alt_weights = []
                        for i, move in enumerate(legal_moves):
                            move_type_check = 'HINT' if move['action_type'].startswith('REVEAL') else move['action_type']
                            if move_type_check == move_type:
                                alt_moves.append(i)
                                # Add situational weight adjustments
                                if move_type_check == 'PLAY':
                                    card_knowledge = observation['card_knowledge'][0][move['card_index']]
                                    if card_knowledge['color'] is not None or card_knowledge['rank'] is not None:
                                        alt_weights.append(1.5)
                                    else:
                                        alt_weights.append(1.0)
                                else:
                                    alt_weights.append(1.0)
                        
                        if alt_moves:
                            # Normalize alt_weights
                            total = sum(alt_weights)
                            alt_weights = [w/total for w in alt_weights]
                            # Choose alternative move
                            alt_index = np.random.choice(alt_moves, p=alt_weights)
                            return legal_moves[alt_index], f"{reasoning}\nPersonality influence: Preferred {move_type} action due to {self.name}'s traits."
                
                return llm_move, reasoning
            
        except Exception as e:
            print(f"API error in action selection: {e}")
            return random.choice(legal_moves), "Fallback random move due to error"

        return random.choice(legal_moves), "Fallback random move"

    def _get_fallback_move(self, observation, legal_moves):
        """Intelligent fallback move selection."""
        fireworks = observation['fireworks']
        known_cards = observation['card_knowledge'][0]
        
        for move in legal_moves:
            if move['action_type'] == 'PLAY':
                card = known_cards[move['card_index']]
                if card['color'] is not None and card['rank'] is not None:
                    if fireworks[card['color']] == card['rank']:
                        return move

        if observation['information_tokens'] == 0:
            discard_moves = [m for m in legal_moves if m['action_type'] == 'DISCARD']
            if discard_moves:
                return random.choice(discard_moves)

        if observation['information_tokens'] >= 8:
            hint_moves = [m for m in legal_moves if m['action_type'].startswith('REVEAL')]
            if hint_moves:
                return random.choice(hint_moves)

        return random.choice(legal_moves)

    def act(self, observation):
        """Enhanced action selection using hybrid approach."""
        print(f"\n{self.name}'s turn:")
        legal_moves = observation['legal_moves']
        
        try:
            # Get strategic analysis
            strategy = self._analyze_game_state(observation)
            print("\nStrategic Analysis:")
            print(strategy)
            
            # Get move recommendation with reasoning
            action, reasoning = self._get_action_recommendation(observation, legal_moves)
            
            print("\nReasoning:")
            print(reasoning)
            
            print("\nChosen Action:")
            print(f"Action type: {action['action_type']}")
            if 'card_index' in action:
                print(f"Card index: {action['card_index']}")
            
            return action
            
        except Exception as e:
            print(f"Error during turn: {e}")
            return super().act(observation)

def format_student_persona(row):
    """Format complete student survey data into a persona description."""
    traits = []
    traits.append(f"age {row['q1']}")
    traits.append(f"{row['q2']}")
    traits.append(f"from a {row['q3']} area")
    traits.append(f"values {row['q5']}")
    traits.append(f"{row['q6']} person")
    traits.append(f"{row['q7']} in nature")
    traits.append(f"enjoys {row['q8']}")
    traits.append(f"interested in {row['q9']}")
    traits.append(f"politically {row['q10']}")
    traits.append(f"has {row['q11']} close friends")
    
    if pd.notna(row['q14']):
        traits.append(f"Myers-Briggs type {row['q14']}")
    
    traits.append(f"seeks {row['q15']}")
    traits.append(f"concerned about {row['q16']}")
    traits.append(f"approaches problems through {row['q21']}")
    traits.append(f"makes decisions by {row['q25']}")
    
    if pd.notna(row['q22']):
        traits.append(f"is {row['q22']}")
    traits.append(f"values {row['q23']}")
    traits.append(f"aspires to {row['q24']}")
    traits.append(f"values {row['q26']} in others")
    
    if pd.notna(row['q29']) and pd.notna(row['q30']):
        traits.append(f"identifies as {row['q29']}, speaks {row['q30']}")
    
    return " ".join(trait for trait in traits if pd.notna(trait) and str(trait) != 'nan')

def select_and_format_players():
    """Select 5 random players from student responses."""
    df = pd.read_csv('cs222_responses.csv')
    selected = df.sample(n=5)
    return [{
        "name": row['sunet'],
        "persona": format_student_persona(row)
    } for _, row in selected.iterrows()]

def create_players(game):
    """Create hybrid agents with random personas."""
    players = select_and_format_players()
    agents = []
    for i, p in enumerate(players):
        agent = HanabiHybridAgent(
            {"game": game, "information_tokens": 8},
            {"name": p["name"], "persona": p["persona"]}
        )
        agent.id = i
        agents.append(agent)
    return agents

def format_game_state(observation, agents):
    """Format the current game state in a clear way."""
    fireworks = observation['fireworks']
    info_tokens = observation['information_tokens']
    life_tokens = observation['life_tokens']
    current_player_id = observation['current_player']
    
    # Format fireworks (stacks)
    stack_status = []
    for color, height in fireworks.items():
        stack_status.append(f"{color}: {height}")
    
    # Format hand information for all players
    hand_info = []
    for player_idx, hand in enumerate(observation['observed_hands']):
        player_name = agents[player_idx].name
        cards_info = []
        
        # If this is the current player, show their cards as hidden but with hints
        if player_idx == current_player_id:
            for card_idx, _ in enumerate(observation['card_knowledge'][0]):
                hints = []
                card_knowledge = observation['card_knowledge'][0][card_idx]
                if card_knowledge['color'] is not None:
                    hints.append(f"color:{card_knowledge['color']}")
                if card_knowledge['rank'] is not None:
                    hints.append(f"rank:{card_knowledge['rank'] + 1}")
                hint_str = f"({', '.join(hints)})" if hints else "(no hints)"
                cards_info.append(f"Card {card_idx}: {hint_str}")
        else:
            # For other players, show their cards and hints
            for card_idx, card in enumerate(hand):
                hints = []
                if card:  # Card might be None if already played
                    card_knowledge = observation['card_knowledge'][player_idx][card_idx]
                    visible_str = f"[{card['color']} {card['rank'] + 1}]"
                    if card_knowledge['color'] is not None:
                        hints.append(f"color:{card_knowledge['color']}")
                    if card_knowledge['rank'] is not None:
                        hints.append(f"rank:{card_knowledge['rank'] + 1}")
                    hint_str = f"({', '.join(hints)})" if hints else "(no hints)"
                    cards_info.append(f"Card {card_idx}: {visible_str} {hint_str}")
                else:
                    cards_info.append(f"Card {card_idx}: (empty)")
        
        hand_info.append(f"{player_name}: {' | '.join(cards_info)}")
    
    state_str = "Current Game State:\n"
    state_str += f"Stacks: {' | '.join(stack_status)}\n"
    state_str += f"Info tokens: {info_tokens} | Life tokens: {life_tokens}\n"
    state_str += "Hands:\n"
    state_str += '\n'.join(hand_info)
    
    return state_str

def run_game():
    """Run a single game with hybrid agents."""
    # Create a timestamp for the log file
    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    log_filename = f'hanabi_game_{timestamp}.txt'
    
    with open(log_filename, 'w') as log_file:
        def log_and_print(message):
            print(message)
            log_file.write(message + '\n')
            log_file.flush()
        
        environment = rl_env.make('Hanabi-Full', num_players=5)
        observations = environment.reset()
        agents = create_players(environment.game)
        
        # Store player names in each agent
        for agent in agents:
            agent.all_player_names = [a.name for a in agents]
        
        # Display initial personas
        log_and_print("\nGame Starting! Meet our players:")
        for agent in agents:
            log_and_print(f"\n{agent.name}'s persona: {agent.persona}")
        log_and_print("\n" + "="*80 + "\n")
        
        done = False
        game_score = 0
        turn_count = 0
        
        while not done:
            for agent_id, agent in enumerate(agents):
                observation = observations['player_observations'][agent_id]
                if observation['current_player'] == agent_id:
                    turn_count += 1
                    log_and_print(f"\nTurn {turn_count}: {agent.name}'s turn")
                    log_and_print(format_game_state(observation, agents))
                    
                    action = agent.act(observation)
                    action_str = f"Action taken: {action['action_type']}"
                    if 'card_index' in action:
                        action_str += f" card {action['card_index']}"
                    elif 'color' in action:
                        action_str += f" hint color {action['color']}"
                    elif 'rank' in action:
                        action_str += f" hint rank {action['rank'] + 1}"
                    log_and_print(action_str)
                    log_and_print("-" * 40)
                    
                    observations, reward, done, _ = environment.step(action)
                    current_score = sum(observations['player_observations'][0]['fireworks'].values())
                    
                    if done:
                        game_score = max(0, current_score)
                        break
        
        final_message = f"\nGame completed! Final score: {game_score}"
        log_and_print(final_message)
        log_and_print(f"\nGame log saved to: {log_filename}")
        
        return game_score

if __name__ == "__main__":
    score = run_game()
