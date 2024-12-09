import openai
import os
from typing import Dict, List, Any

class LLMStrategist:
    def __init__(self, api_key: str = None):
        """Initialize with optional API key or use environment variable."""
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        openai.api_key = self.api_key
        
    def analyze_game_state(self, observation: Dict, player_name: str, personality: str) -> str:
        """Generate strategic analysis using GPT-4."""
        prompt = f"""As an expert Hanabi player advising {player_name} who {personality}, analyze this game state:
        
        Fireworks: {observation['fireworks']}
        Information tokens: {observation['information_tokens']}
        Life tokens: {observation['life_tokens']}
        
        Consider:
        1. Current priority (playing cards vs giving hints vs discarding)
        2. Risk assessment given remaining life tokens
        3. Team coordination needs
        4. Critical cards that must be saved
        
        Provide strategic advice in 3-4 sentences."""
        
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert Hanabi strategy advisor."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=150,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"API error: {e}")
            return self._get_fallback_analysis(observation)
    
    def evaluate_hint_value(self, hint_move: Dict, observation: Dict, target_name: str) -> str:
        """Evaluate the potential value of a hint."""
        affected_cards = self._get_hint_affected_cards(hint_move, observation)
        
        prompt = f"""As a Hanabi expert, evaluate this potential hint to {target_name}:
        
        Hint type: {hint_move['action_type']}
        Affected cards: {affected_cards}
        Game state: {observation['fireworks']}
        
        Consider:
        1. Immediate playability
        2. Future value
        3. Prevention of discards
        
        Provide hint evaluation in 2-3 sentences."""
        
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert Hanabi strategy advisor."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=100,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"API error: {e}")
            return self._get_fallback_hint_evaluation(hint_move, observation)

    def _get_hint_affected_cards(self, hint_move: Dict, observation: Dict) -> List[Dict]:
        """Get details of cards affected by a hint."""
        target_hand = observation['observed_hands'][hint_move['target_offset']]
        if hint_move['action_type'] == 'REVEAL_COLOR':
            return [
                {'position': i, 'color': card['color'], 'rank': card['rank']}
                for i, card in enumerate(target_hand)
                if card['color'] == hint_move['color']
            ]
        else:  # REVEAL_RANK
            return [
                {'position': i, 'color': card['color'], 'rank': card['rank']}
                for i, card in enumerate(target_hand)
                if card['rank'] == hint_move['rank']
            ]

    def _get_fallback_analysis(self, observation: Dict) -> str:
        """Provide basic analysis when API is unavailable."""
        life_tokens = observation['life_tokens']
        info_tokens = observation['information_tokens']
        
        if life_tokens == 1:
            return "Critical situation with one life remaining. Focus on safe plays and information gathering."
        elif info_tokens == 0:
            return "No information tokens left. Must discard to enable team coordination."
        else:
            return "Standard situation. Balance information sharing with careful progress."

class EnhancedPersonaAgent(PersonaAgent):
    def __init__(self, config, agent_data):
        super().__init__(config, agent_data)
        self.strategist = LLMStrategist()
        self.last_analysis = None
        self.analysis_frequency = 5  # Only get full analysis every N turns
        
    def act(self, observation):
        self.turns_played += 1
        print(f"\n{self.name}'s turn:")
        
        # Get strategic analysis occasionally or in critical situations
        if (self.turns_played % self.analysis_frequency == 0 or 
            observation['life_tokens'] == 1 or 
            observation['information_tokens'] == 0):
            self.last_analysis = self.strategist.analyze_game_state(
                observation, self.name, self.persona
            )
            print("\nStrategic Analysis:")
            print(self.last_analysis)
        
        # Regular hint analysis with LLM enhancement
        hint_analysis = self._analyze_received_hints(observation)
        print("\nAnalyzing received hints:")
        for analysis in hint_analysis:
            if analysis['known_info']:
                self._print_enhanced_hint_analysis(analysis, observation)
        
        # Use regular logic for must-play rules
        must_play = self._evaluate_must_play_rules(observation)
        if must_play and must_play in observation['legal_moves']:
            return must_play
        
        # Get valid moves considering must-save rules
        legal_moves = self._get_valid_moves(observation)
        
        # Enhanced hint selection with LLM evaluation
        if any(m['action_type'].startswith('REVEAL') for m in legal_moves):
            hint_move = self._select_enhanced_hint(legal_moves, observation)
            if hint_move:
                return hint_move
        
        # Enhanced play/discard decision
        return self._make_enhanced_decision(observation, legal_moves, hint_analysis)
    
    def _print_enhanced_hint_analysis(self, analysis: Dict, observation: Dict):
        """Print detailed analysis of received hints with LLM insights."""
        print(f"\nCard {analysis['card_index']}:")
        print("Known information:")
        for info in analysis['known_info']:
            print(f"  * {info}")
            
        if analysis['known_info'][0] != "No hints received yet":
            hint_value = self.strategist.evaluate_hint_value(
                {'action_type': 'ANALYZE', 'card_index': analysis['card_index']},
                observation,
                self.name
            )
            print("\nHint evaluation:")
            print(hint_value)
        
        print(f"Playability: {analysis['playability']}")
        print(f"Confidence: {analysis['confidence']:.2f}")
        
    def _select_enhanced_hint(self, legal_moves: List[Dict], observation: Dict) -> Dict:
        """Select best hint using LLM evaluation."""
        hint_moves = [m for m in legal_moves if m['action_type'].startswith('REVEAL')]
        if not hint_moves:
            return None
            
        best_hint = None
        best_value = ""
        
        for move in hint_moves[:3]:  # Limit API calls
            target_id = (observation['current_player'] + move['target_offset']) % len(self.all_player_names)
            target_name = self.all_player_names[target_id]
            
            hint_value = self.strategist.evaluate_hint_value(move, observation, target_name)
            
            if "immediate" in hint_value.lower() or "critical" in hint_value.lower():
                return move  # Return immediately valuable hints
            elif not best_hint:
                best_hint = move
                best_value = hint_value
                
        return best_hint
    
    def _make_enhanced_decision(self, observation: Dict, legal_moves: List[Dict], hint_analysis: List[Dict]) -> Dict:
        """Make final decision incorporating LLM analysis."""
        weights = self._evaluate_persona_rules(observation, legal_moves)
        risk_tolerance = self._calculate_current_risk_tolerance(observation)
        
        # Modify weights based on LLM analysis if available
        if self.last_analysis:
            if "critical" in self.last_analysis.lower():
                weights['PLAY'] *= 0.5
            if "safe" in self.last_analysis.lower():
                weights['HINT'] *= 1.5
            if "must discard" in self.last_analysis.lower():
                weights['DISCARD'] *= 2.0
        
        # Calculate final move weights
        move_weights = []
        for move in legal_moves:
            weight = self._calculate_move_weight(move, weights, hint_analysis, risk_tolerance)
            move_weights.append(max(0.01, weight))
        
        # Normalize and select move
        move_weights = np.array(move_weights)
        move_weights = move_weights / move_weights.sum()
        
        chosen_idx = np.random.choice(len(legal_moves), p=move_weights)
        return legal_moves[chosen_idx]

def create_players(game):
    """Creates players with enhanced LLM capabilities."""
    players = select_and_format_players()
    agents = []
    for i, p in enumerate(players):
        agent = EnhancedPersonaAgent(
            {"game": game, "information_tokens": 8},
            {"name": p["name"], "persona": p["persona"]}
        )
        agent.id = i
        agent.all_player_names = [p["name"] for p in players]
        agents.append(agent)
    return agents