from openai import OpenAI
from settings import OPENAI_API_KEY, LLM_VERS, DEBUG
import os
import random
from hanabi_learning_environment import rl_env
from hanabi_learning_environment.agents.simple_agent import SimpleAgent
from agents import agent_list  # Import personas
import pandas as pd
import numpy as np

# Create a single client instance to be shared
try:
    client = OpenAI(api_key=OPENAI_API_KEY)
except Exception as e:
    print(f"Warning: Could not initialize OpenAI client: {e}")
    client = None

class HanabiLLMAgent(SimpleAgent):
    def __init__(self, config, agent_data):
        super().__init__(config)
        self.name = agent_data["name"]
        self.persona = agent_data["persona"]
        self.all_player_names = []
        self.client = client  # Use the shared client instance
        
    def get_player_name(self, current_player, target_offset):
        """Get the name of a player based on current player and target offset."""
        if not self.all_player_names:
            return f"Player {target_offset}"  # Fallback if names aren't set
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

    def _get_fallback_move(self, observation, legal_moves):
        """Intelligent fallback move selection."""
        # First check for known playable cards
        fireworks = observation['fireworks']
        known_cards = observation['card_knowledge'][0]
        
        # Look for definitely playable cards
        for move in legal_moves:
            if move['action_type'] == 'PLAY':
                card = known_cards[move['card_index']]
                if card['color'] and card['rank'] is not None:
                    if fireworks[card['color']] == card['rank']:
                        return move

        # If we have no information tokens, must discard
        if observation['information_tokens'] == 0:
            discard_moves = [m for m in legal_moves if m['action_type'] == 'DISCARD']
            if discard_moves:
                return random.choice(discard_moves)

        # If we have max information tokens, prefer giving hints
        if observation['information_tokens'] >= 8:
            hint_moves = [m for m in legal_moves if m['action_type'].startswith('REVEAL')]
            if hint_moves:
                return random.choice(hint_moves)

        # Default to random legal move
        return random.choice(legal_moves)

    def _parse_move_recommendation(self, response, legal_moves):
        """Parse LLM move recommendation."""
        try:
            move_line = [line for line in response.split('\n') if line.startswith('MOVE:')][0]
            move_index = int(move_line.split(':')[1].strip())
            if 0 <= move_index < len(legal_moves):
                return legal_moves[move_index]
        except:
            if DEBUG:
                print("Failed to parse LLM move recommendation")
        return self._get_fallback_move(observation, legal_moves)

    def _get_fallback_analysis(self, observation):
        """Generate fallback analysis when LLM is unavailable."""
        life_tokens = observation['life_tokens']
        info_tokens = observation['information_tokens']
        fireworks = observation['fireworks']
        
        analysis = []
        
        # Life token assessment
        if life_tokens == 1:
            analysis.append("Critical situation with one life remaining. Must play very safely.")
        elif life_tokens == 2:
            analysis.append("Need to be cautious with plays.")
        
        # Information token assessment
        if info_tokens == 0:
            analysis.append("Must discard to regain information tokens.")
        elif info_tokens == 8:
            analysis.append("Should use information tokens for team coordination.")
        
        # Game progress assessment
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

    def _get_card_info(self, observation, card_index):
        """Get known information about a card."""
        card_knowledge = observation['card_knowledge'][0][card_index]
        info = []
        if card_knowledge['color'] is not None:
            info.append(f"color: {card_knowledge['color']}")
        if card_knowledge['rank'] is not None:
            info.append(f"rank: {card_knowledge['rank'] + 1}")
        return f"({', '.join(info)})" if info else "(no information)"

    def _evaluate_playability(self, card, fireworks):
        """Check if a card would be playable based on current fireworks."""
        if card['color'] is None or card['rank'] is None:
            return False
        return fireworks[card['color']] == card['rank']

    def _is_potentially_playable(self, card_knowledge, fireworks):
        """Evaluate if a card could be playable based on partial information."""
        if card_knowledge['color'] is not None and card_knowledge['rank'] is not None:
            return fireworks[card_knowledge['color']] == card_knowledge['rank']
        elif card_knowledge['rank'] is not None:
            # If we only know rank, check if that rank is needed in any stack
            return any(height == card_knowledge['rank'] for height in fireworks.values())
        elif card_knowledge['color'] is not None:
            # If we only know color, check if that color's stack needs cards
            return fireworks[card_knowledge['color']] < 5
        return False

    def _get_action_recommendation(self, observation, legal_moves):
        """Get LLM recommendation with bias towards hinted cards and useful hints."""
        if not self.client:
            return self._get_fallback_move(observation, legal_moves)

        fireworks = observation['fireworks']
        
        # First priority: Play cards we have hints for and believe are playable
        for move in legal_moves:
            if move['action_type'] == 'PLAY':
                card_knowledge = observation['card_knowledge'][0][move['card_index']]
                if self._is_potentially_playable(card_knowledge, fireworks):
                    if card_knowledge['color'] is not None or card_knowledge['rank'] is not None:
                        return move

        # Second priority: Give hints about playable cards if we have information tokens
        if observation['information_tokens'] > 0:
            for move in legal_moves:
                if move['action_type'].startswith('REVEAL'):
                    target_offset = move['target_offset']
                    target_hand = observation['observed_hands'][target_offset]
                    target_knowledge = observation['card_knowledge'][target_offset]
                    
                    # Check if this hint would identify a playable card
                    for card_idx, (card, knowledge) in enumerate(zip(target_hand, target_knowledge)):
                        if card is not None:  # Card exists
                            if move['action_type'] == 'REVEAL_COLOR' and move['color'] == card['color']:
                                if fireworks[card['color']] == card['rank']:
                                    return move
                            elif move['action_type'] == 'REVEAL_RANK' and move['rank'] == card['rank']:
                                for color, height in fireworks.items():
                                    if height == card['rank']:
                                        return move

        # Third priority: Play cards with at least some hints if we're in a good position
        if observation['life_tokens'] > 1:
            for move in legal_moves:
                if move['action_type'] == 'PLAY':
                    card_knowledge = observation['card_knowledge'][0][move['card_index']]
                    if card_knowledge['color'] is not None or card_knowledge['rank'] is not None:
                        return move

        # Fourth priority: Discard cards with no hints if we need information tokens
        if observation['information_tokens'] < 8:
            for move in legal_moves:
                if move['action_type'] == 'DISCARD':
                    card_knowledge = observation['card_knowledge'][0][move['card_index']]
                    if card_knowledge['color'] is None and card_knowledge['rank'] is None:
                        return move

        # Finally, fall back to the existing LLM logic with the context of our strategy
        prompt = f"""As {self.name}, considering your personality who {self.persona}, analyze these possible moves in Hanabi:
    Game State:
    - Fireworks: {observation['fireworks']}
    - Info tokens: {observation['information_tokens']}
    - Life tokens: {observation['life_tokens']}
    Legal Moves:
    {self._format_legal_moves(legal_moves, observation)}

    Prioritize:
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
                max_tokens=100,
                temperature=0.7
            )
            return self._parse_move_recommendation(response.choices[0].message.content, legal_moves)
        except Exception as e:
            print(f"API error in action selection: {e}")
            return random.choice(legal_moves)
    

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
                formatted.append(f"{i}: Play card {move['card_index']}")
            elif move['action_type'] == 'DISCARD':
                formatted.append(f"{i}: Discard card {move['card_index']}")
            elif move['action_type'].startswith('REVEAL'):
                target_offset = move['target_offset']
                target_name = self.get_player_name(observation['current_player'], target_offset)
                info_type = 'color' if move['action_type'] == 'REVEAL_COLOR' else 'rank'
                info_value = move['color'] if info_type == 'color' else move['rank'] + 1
                formatted.append(f"{i}: Tell {target_name} about {info_type} {info_value}")
        return "\n".join(formatted)
        
    def _parse_move_recommendation(self, response, legal_moves):
        """Parse LLM move recommendation."""
        try:
            move_line = [line for line in response.split('\n') if line.startswith('MOVE:')][0]
            move_index = int(move_line.split(':')[1].strip())
            if 0 <= move_index < len(legal_moves):
                return legal_moves[move_index]
        except:
            pass
        return random.choice(legal_moves)
        
    def _get_fallback_analysis(self, observation):
        """Provide basic analysis when API fails."""
        life_tokens = observation['life_tokens']
        info_tokens = observation['information_tokens']
        
        if life_tokens == 1:
            return "Critical situation with one life remaining. Must play very safely."
        elif info_tokens == 0:
            return "No information tokens left. Need to discard to enable communication."
        else:
            return "Standard situation. Focus on efficient information sharing."
            
    def act(self, observation):
        """Enhanced action selection using LLM."""
        print(f"\n{self.name}'s turn:")
        legal_moves = observation['legal_moves']
        
        # Check if API is available
        if not self.client:
            print("Warning: OpenAI client not available, using fallback logic")
            return super().act(observation)
        
        try:
            # Get strategic analysis
            strategy = self._analyze_game_state(observation)
            print("\nStrategic Analysis:")
            print(strategy)
            
            # Get move recommendation
            action = self._get_action_recommendation(observation, legal_moves)
            
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
    
    # Demographics and background
    traits.append(f"age {row['q1']}")
    traits.append(f"{row['q2']}")
    traits.append(f"from a {row['q3']} area")
    
    # Values and priorities
    traits.append(f"values {row['q5']}")
    
    # Personality and approach
    traits.append(f"{row['q6']} person")
    traits.append(f"{row['q7']} in nature")
    traits.append(f"enjoys {row['q8']}")
    traits.append(f"interested in {row['q9']}")
    
    # Social and political views
    traits.append(f"politically {row['q10']}")
    traits.append(f"has {row['q11']} close friends")
    
    # Personal characteristics
    if pd.notna(row['q14']):
        traits.append(f"Myers-Briggs type {row['q14']}")
    
    # Goals and concerns
    traits.append(f"seeks {row['q15']}")
    traits.append(f"concerned about {row['q16']}")
    
    # Decision making and problem solving
    traits.append(f"approaches problems through {row['q21']}")
    traits.append(f"makes decisions by {row['q25']}")
    
    # Values and background
    if pd.notna(row['q22']):
        traits.append(f"is {row['q22']}")
    traits.append(f"values {row['q23']}")
    traits.append(f"aspires to {row['q24']}")
    traits.append(f"values {row['q26']} in others")
    
    # Cultural background
    if pd.notna(row['q29']) and pd.notna(row['q30']):
        traits.append(f"identifies as {row['q29']}, speaks {row['q30']}")
    
    return " ".join(trait for trait in traits if pd.notna(trait) and str(trait) != 'nan')

def select_and_format_players():
    """Select 5 random players from student responses."""
    df = pd.read_csv('cs222_responses.csv')
    selected = df.sample(n=5)
    
    return [
        {
            "name": row['sunet'],
            "persona": format_student_persona(row)
        }
        for _, row in selected.iterrows()
    ]

def create_players(game):
    """Create LLM-enhanced Hanabi players."""
    players = select_and_format_players()
    agents = []
    for i, p in enumerate(players):
        agent = HanabiLLMAgent(
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
    """Runs a single game with improved state display and logs to file."""
    # Create a timestamp for the log file
    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    log_filename = f'hanabi_game_{timestamp}.txt'
    
    with open(log_filename, 'w') as log_file:
        def log_and_print(message):
            print(message)
            log_file.write(message + '\n')
            log_file.flush()  # Ensure immediate writing to file
        
        environment = rl_env.make('Hanabi-Full', num_players=5)
        observations = environment.reset()
        agents = create_players(environment.game)
        
        # Store player names in each agent instance
        for i, agent in enumerate(agents):
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