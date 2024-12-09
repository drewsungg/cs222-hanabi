from hanabi_learning_environment import rl_env
from hanabi_learning_environment.agents.simple_agent import SimpleAgent
from agents import agent_list
import numpy as np
import pandas as pd

# Define player archetypes and their action weights
PLAYER_ARCHETYPES = {
    "Conservative": {
        "traits": ["thoughtful", "analytical", "careful"],
        "action_weights": {
            "PLAY": 0.2,
            "DISCARD": 0.3, 
            "HINT": 1.0
        }
    },
    "Risk-Taker": {
        "traits": ["confident", "ambitious"],
        "action_weights": {
            "PLAY": 1.0,
            "DISCARD": 0.3,
            "HINT": 0.2
        }
    },
    "Team-Player": {
        "traits": ["compassion", "empathy", "helping others"],
        "action_weights": {
            "PLAY": 0.5,
            "DISCARD": 0.5,
            "HINT": 0.5
        }
    },
}

class PersonaWeightAgent(SimpleAgent):
    def __init__(self, config, agent_data):
        super().__init__(config)
        self.name = agent_data["name"]
        self.persona = agent_data["persona"]
        self.archetype = self.persona
            
    def _weight_moves(self, observation, legal_moves):
        """Apply dynamic weights based on game state and hints."""
        weights = PLAYER_ARCHETYPES[self.archetype]["action_weights"].copy()
        move_weights = []
        
        # Get known card information
        known_cards = observation['card_knowledge'][0]
        info_tokens = observation['information_tokens']
        life_tokens = observation['life_tokens']
        
        for move in legal_moves:
            base_weight = weights.get(move['action_type'], 0.3)
            
            # Dynamic weight adjustments
            if move['action_type'] == 'PLAY':
                card_index = move.get('card_index')
                if card_index is not None and known_cards[card_index]['color']:
                    base_weight *= 1.5  # Boost confidence for known cards
                elif life_tokens == 1:
                    base_weight *= 0.5  # More cautious on last life
                    
            elif move['action_type'] == 'REVEAL_COLOR':
                if info_tokens == 0:
                    base_weight = 0  # Can't hint without tokens
                elif info_tokens >= 7:
                    base_weight *= 1.3  # Encourage using excess tokens
                    
            elif move['action_type'] == 'DISCARD':
                if info_tokens == 0:
                    base_weight *= 1.5  # Need tokens
                elif info_tokens >= 7:
                    base_weight *= 0.5  # Don't waste cards
            
            move_weights.append(base_weight)
            
        # Normalize weights
        move_weights = np.array(move_weights)
        if move_weights.sum() > 0:
            move_weights = move_weights / move_weights.sum()
        return move_weights

    def act(self, observation):
        print(f"\n{self.name}'s turn:")
        print(f"You are an agent who {self.persona}")
        
        # Game state analysis
        fireworks = observation['fireworks']
        info_tokens = observation['information_tokens']
        life_tokens = observation['life_tokens']
        legal_moves = observation['legal_moves']
        
        print("\nMy cards:")
        print(self._analyze_known_cards(observation))
        
        print("\nThinking process:")
        print(self._generate_response(observation, legal_moves))
        
        # Get weighted action
        move_weights = self._weight_moves(observation, legal_moves)
        chosen_move_idx = np.random.choice(len(legal_moves), p=move_weights)
        action = legal_moves[chosen_move_idx]
        
        print("\nReasoning:")
        print(self._generate_decision_reasoning(action, observation))
        
        return action
    
    def _analyze_known_cards(self, observation):
        """Analyze cards that have been hinted at in hand."""
        known_cards = observation['card_knowledge'][0]
        hand = observation['observed_hands'][0]
        
        card_analysis = []
        for i, (card, knowledge) in enumerate(zip(hand, known_cards)):
            if knowledge['color'] is not None or knowledge['rank'] is not None:
                hint_info = []
                if knowledge['color']:
                    hint_info.append(f"color {knowledge['color']}")
                if knowledge['rank'] is not None:
                    hint_info.append(f"rank {knowledge['rank'] + 1}")
                card_analysis.append(f"Card {i}: Known {' and '.join(hint_info)}")
        
        return "\n- ".join(["My hinted cards:"] + card_analysis) if card_analysis else "No hints received about my cards yet."

    def _generate_response(self, observation, legal_moves):
        """Generate strategic reasoning with dynamic analysis."""
        fireworks = observation['fireworks']
        info_tokens = observation['information_tokens']
        life_tokens = observation['life_tokens']
        total_score = sum(fireworks.values())
        max_possible = len(fireworks.keys()) * 5
        
        analysis = [
            f"Currently at {total_score}/{max_possible} points.",
            f"\nAnalyzing our position:",
            f"- Life tokens ({life_tokens}/3): {'Critical' if life_tokens==1 else 'Concerning' if life_tokens==2 else 'Safe'}",
            f"- Info tokens ({info_tokens}/8): {'Must discard' if info_tokens==0 else 'Limited' if info_tokens<3 else 'Sufficient'}",
            "\nStack progress:",
        ]
        
        for color, value in fireworks.items():
            status = "Complete!" if value==5 else "Near completion" if value>=3 else "In progress" if value>0 else "Not started"
            analysis.append(f"- {color}: {value}/5 ({status})")
        
        analysis.append("\nConsidering options...")
        return "\n".join(analysis)

    def _generate_decision_reasoning(self, action, observation):
        """Generate natural reasoning about decision and target player."""
        game_state = f"With {observation['life_tokens']} lives and {observation['information_tokens']} info tokens"
        fireworks_state = f"fireworks at {observation['fireworks']}"
        
        # For hints
        if action['action_type'] == 'REVEAL_COLOR':
            target_player = agent_list[action['target_offset']]['name']
            return f"I noticed {target_player} has important {action['color']} cards they should know about. My hint should help them make a good decision on their turn."
        
        # For plays
        elif action['action_type'] == 'PLAY':
            card_idx = action.get('card_index', 0)  # Default to 0 if not found
            known_info = observation['card_knowledge'][0][card_idx]
            if known_info['color'] or known_info['rank'] is not None:
                return f"{game_state}, based on the hints I've received about my {card_idx}th card, I believe it's a good time to play it."
            else:
                return f"{game_state}, while I don't have hints about this card, I think playing my {card_idx}th card is worth the risk."
        
        # For discards
        elif action['action_type'] == 'DISCARD':
            card_idx = action.get('card_index', 0)  # Default to 0 if not found
            return f"{game_state}, I'm choosing to discard my {card_idx}th card to give us more flexibility with our information tokens."
        
        # Default case
        else:
            return f"{game_state}, I'm taking this action to advance our {fireworks_state}"
    
def analyze_student_traits(student_data):
    """Analyze full student profile to determine archetype."""
    traits = {
        "Conservative": 0,
        "Risk-Taker": 0,
        "Team-Player": 0
    }
    
    # Personality indicators (q6, q7)
    if any(word in str(student_data['q6']).lower() for word in ['analytical', 'thoughtful', 'introverted']):
        traits["Conservative"] += 1
    if any(word in str(student_data['q6']).lower() for word in ['extroverted', 'outgoing']):
        traits["Risk-Taker"] += 1
    
    # Decision making (q21)
    if 'logical analysis' in str(student_data['q21']).lower():
        traits["Conservative"] += 1
    if 'intuition' in str(student_data['q21']).lower():
        traits["Risk-Taker"] += 1
    
    # Life priorities (q5)
    if 'compassion' in str(student_data['q5']).lower():
        traits["Team-Player"] += 1
    if 'ambition' in str(student_data['q5']).lower():
        traits["Risk-Taker"] += 1
    
    # Activities (q8)
    if 'helping others' in str(student_data['q8']).lower():
        traits["Team-Player"] += 2
    if 'personal projects' in str(student_data['q8']).lower():
        traits["Conservative"] += 1
    
    # Decision style (q25)
    if 'analyzing all options' in str(student_data['q25']).lower():
        traits["Conservative"] += 1
    if 'immediate action' in str(student_data['q25']).lower():
        traits["Risk-Taker"] += 1
    
    # Get archetype with highest score
    return max(traits.items(), key=lambda x: x[1])[0]

def select_and_categorize_players():
    """Select 5 random players from survey and categorize them."""
    df = pd.read_csv('cs222_responses.csv')
    players = df.sample(n=5).to_dict('records')
    
    categorized_players = []
    for player in players:
        archetype = analyze_student_traits(player)
        categorized_players.append({
            "name": player['sunet'],
            "persona": archetype
        })
    
    return categorized_players

def create_players(game):
    """Creates players with random personas from survey data."""
    players = select_and_categorize_players()
    return [PersonaWeightAgent(
        {"game": game, "information_tokens": 8},
        {"name": p["name"], "persona": p["persona"]}
    ) for p in players]

def run_game():
    """Runs a single game and returns the score."""
    environment = rl_env.make('Hanabi-Full', num_players=5)
    observations = environment.reset()
    
    agents = create_players(environment.game)
    done = False
    game_score = 0
    turn_count = 0
    
    while not done:
        for agent_id, agent in enumerate(agents):
            observation = observations['player_observations'][agent_id]
            legal_moves = observation['legal_moves']
            
            if observation['current_player'] == agent_id:
                turn_count += 1
                action = agent.act(observation)
                
                print(f"\nTurn {turn_count}: {agent.name}")
                print(f"Action chosen: {action}")
                
                observations, reward, done, _ = environment.step(action)
                current_score = sum(observation['fireworks'].values())
                
                if done:
                    game_score = current_score
                    break
    
    print(f"\nGame completed. Score: {game_score}")
    return game_score

if __name__ == "__main__":
    score = run_game()