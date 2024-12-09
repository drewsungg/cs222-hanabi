from hanabi_learning_environment import rl_env
from hanabi_learning_environment.agents.simple_agent import SimpleAgent
from agents import agent_list  # Import personas

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

class PersonaAgent(SimpleAgent):
    def __init__(self, config, agent_data):
        super().__init__(config)
        self.name = agent_data["name"]
        self.persona = agent_data["persona"]

    def _analyze_known_cards(self, observation):
        """Analyze cards that have been hinted at in hand."""
        known_cards = observation['card_knowledge'][0]  # Current player's hand knowledge
        hand = observation['observed_hands'][0]  # Current player's hand
        
        card_analysis = []
        for i, (card, knowledge) in enumerate(zip(hand, known_cards)):
            if knowledge['color'] is not None or knowledge['rank'] is not None:
                hint_info = []
                if knowledge['color']:
                    hint_info.append(f"color {knowledge['color']}")
                if knowledge['rank'] is not None:  # rank can be 0
                    hint_info.append(f"rank {knowledge['rank'] + 1}")
                card_analysis.append(f"Card {i}: Known {' and '.join(hint_info)}")
        
        if not card_analysis:
            return "No hints received about my cards yet."
        return "My hinted cards:\n- " + "\n- ".join(card_analysis)

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

        # Analyze known cards
        known_cards = observation['card_knowledge'][0]
        card_index = action.get('card_index', None)
        card_knowledge = known_cards[card_index] if card_index is not None else None

        # For hints
        if action['action_type'] == 'REVEAL_COLOR':
            target_player = agent_list[action['target_offset']]['name']
            return f"I noticed {target_player} has important {action['color']} cards they should know about. My hint should help them make a good decision on their turn."

        # For plays - consider hints received
        elif action['action_type'] == 'PLAY':
            hint_info = []
            if card_knowledge and card_knowledge['color']:
                hint_info.append(f"color {card_knowledge['color']}")
            if card_knowledge and card_knowledge['rank'] is not None:
                hint_info.append(f"rank {card_knowledge['rank'] + 1}")
                
            if hint_info:
                return f"{game_state}, based on hints telling me this is a {' and '.join(hint_info)} card, I'm confident about playing my {card_index}th card. This should help advance our {fireworks_state}."
            else:
                return f"{game_state}, even without hints, I feel this is a good time to play my {card_index}th card to advance our {fireworks_state}."

        # For discards - consider known cards
        else:
            if card_knowledge and (card_knowledge['color'] or card_knowledge['rank'] is not None):
                return f"{game_state}, although I have hints about my {card_index}th card, I'm choosing to discard it since we need more info tokens for coordination."
            else:
                return f"{game_state}, without any hints about my {card_index}th card, I'm opting to discard it for an additional info token."

    def act(self, observation):
            print(f"\n{self.name}'s turn:")
            print("\nRole context:")
            print(f"You are an agent who {self.persona}")
            
            # Game state analysis
            fireworks = observation['fireworks']
            info_tokens = observation['information_tokens']
            life_tokens = observation['life_tokens']
            legal_moves = observation['legal_moves']

             # Add card knowledge analysis
            print("\nMy cards:")
            print(self._analyze_known_cards(observation))
    
            print("\nPrompt:")
            print("Looking at this Hanabi game state:")
            print(f"- Current firework stacks: {fireworks}")
            print(f"- Available info tokens: {info_tokens}")
            print(f"- Remaining life tokens: {life_tokens}")
            print("\nConsidering your background and expertise, what do you think is the optimal legal move?")
            print("Remember to consider:")
            print("- The current game progress")
            print("- Your teammates' personas and likely strategies")
            print("- The risk/reward balance of your potential moves")

            # Generate and print reasoning
            print("\nThinking process:")
            reasoning = self._generate_response(observation, legal_moves)
            print(reasoning)
            
            action = super().act(observation)
            
            print("\nReasoning:")
            print(self._generate_decision_reasoning(action, observation))
            
            print("\nDecision:")
            if action['action_type'] == 'PLAY':
                print(f"Playing card {action['card_index']}")
            elif action['action_type'] == 'REVEAL_COLOR':
                target_name = agent_list[action['target_offset']]['name']
                print(f"Hinting about {action['color']} cards to {target_name}")
            elif action['action_type'] == 'DISCARD':
                print(f"Discarding card {action['card_index']}")
            
            return action

def create_players(game):
    return [PersonaAgent({"game": game, "information_tokens": 8}, agent_data) 
            for agent_data in agent_list]

def run_game():
    # Create environment with just required parameters
    environment = rl_env.make(
        environment_name='Hanabi-Full',
        num_players=5
    )
    
    # Create players
    agents = create_players(environment.game)

    # Run game
    observations = environment.reset()
    done = False
    game_score = 0
    turn_count = 0

    while not done:
        for agent_id, agent in enumerate(agents):
            observation = observations['player_observations'][agent_id]
            legal_moves = observation['legal_moves']
            
            if observation['current_player'] == agent_id:
                turn_count += 1
                
                # Get action and enforce legal moves
                action = agent.act(observation)
                if action not in legal_moves:
                    print(f"Warning: Agent {agent.name} attempted illegal move {action}")
                    print(f"Legal moves: {legal_moves}")
                    action = np.random.choice(legal_moves)
                
                print(f"\nTurn {turn_count}: {agent.name}")
                print(f"Action chosen: {action}")
                print(f"Lives remaining: {observation['life_tokens']}")
                
                observations, reward, done, info = environment.step(action)
                
                # Track current score
                current_score = sum(observation['fireworks'].values())
                print(f"Current fireworks score: {current_score}")
                
                if done:
                    # Use final fireworks state for score instead of reward
                    final_score = sum(observation['fireworks'].values())
                    game_score = final_score
                    print(f"Game ended with score: {game_score}")
                    break

    print(f"\nGame completed. Score: {game_score}")
    return game_score

if __name__ == "__main__":
    score = run_game()