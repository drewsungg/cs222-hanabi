import random
from hanabi_learning_environment import rl_env
from hanabi_learning_environment.agents.simple_agent import SimpleAgent
from agents import agent_list  # Import personas
import pandas as pd
import numpy as np

class PersonaAgent(SimpleAgent):
    def __init__(self, config, agent_data):
        super().__init__(config)
        self.name = agent_data["name"]
        self.persona = agent_data["persona"]
        self.personality_traits = self._extract_personality_traits()
        # Track turn count for risk adjustment
        self.turns_played = 0
        self.base_risk_tolerance = 0.8  # Start aggressive but later get more conservative
            
    def _calculate_current_risk_tolerance(self, observation):
        """Calculate dynamic risk tolerance based on game state and progression."""
        # Base risk tolerance decreases as game progresses
        base_risk = max(0.2, self.base_risk_tolerance * (1 - (self.turns_played / 40)))  # Assuming ~40 turns per game
        
        # Adjust based on life tokens
        life_factor = observation['life_tokens'] / 3.0
        
        # Adjust based on current score vs potential
        current_score = sum(observation['fireworks'].values())
        potential_score = 5 * len(observation['fireworks'])
        score_factor = 1 - (current_score / potential_score)  # More aggressive when score is low
        
        # Personality influence
        personality_modifier = 0.2 if self.personality_traits['risk_taking'] else -0.1
        
        final_risk = base_risk * life_factor * score_factor + personality_modifier
        return max(0.1, min(0.9, final_risk))  # Clamp between 0.1 and 0.9
    
    def _analyze_hint_intention(self, observation, card_index, hinting_player_name=None):
        """Analyze why a specific card might have been hinted at."""
        fireworks = observation['fireworks']
        card_knowledge = observation['card_knowledge'][0][card_index]
        stack_heights = fireworks
        analysis = {
            'intention': 'unknown',
            'confidence': 0.0,
            'reasoning': []
        }
        
        # If we know both color and rank
        if card_knowledge['color'] and card_knowledge['rank'] is not None:
            value = card_knowledge['rank'] + 1
            color = card_knowledge['color']
            stack_height = stack_heights[color]
            
            if stack_height + 1 == value:
                analysis['intention'] = 'immediate_play'
                analysis['confidence'] = 1.0
                analysis['reasoning'].append(f"This {value} {color} is immediately playable on the {color} stack at {stack_height}")
            elif value <= stack_height:
                analysis['intention'] = 'warning'
                analysis['confidence'] = 1.0
                analysis['reasoning'].append(f"This {value} {color} is already played (stack at {stack_height})")
            elif value > stack_height + 1:
                analysis['intention'] = 'save'
                analysis['confidence'] = 0.9
                analysis['reasoning'].append(f"This {value} {color} will be needed later (stack at {stack_height})")
                
        # If we only know the rank
        elif card_knowledge['rank'] is not None:
            value = card_knowledge['rank'] + 1
            if value == 1 and any(h == 0 for h in stack_heights.values()):
                analysis['intention'] = 'immediate_play'
                analysis['confidence'] = 0.9
                analysis['reasoning'].append(f"This is a 1 and some stacks need 1s")
            elif value == 5:
                analysis['intention'] = 'save'
                analysis['confidence'] = 0.8
                analysis['reasoning'].append("This is a 5, which should be saved")
            else:
                playable_in = [color for color, height in stack_heights.items() if height + 1 == value]
                if playable_in:
                    analysis['intention'] = 'potential_play'
                    analysis['confidence'] = 0.7
                    analysis['reasoning'].append(f"This {value} could be playable in {playable_in}")
                    
        # If we only know the color
        elif card_knowledge['color']:
            color = card_knowledge['color']
            stack_height = stack_heights[color]
            if stack_height == 4:
                analysis['intention'] = 'potential_play'
                analysis['confidence'] = 0.8
                analysis['reasoning'].append(f"This {color} card might be the 5 needed")
            else:
                analysis['intention'] = 'information'
                analysis['confidence'] = 0.5
                analysis['reasoning'].append(f"The {color} stack is at {stack_height}, this might be relevant soon")
                
        return analysis

    def _evaluate_persona_rules(self, observation, legal_moves):
        """Apply personality-based rules with dynamic risk adjustment."""
        hint_prefs = self._get_hint_preferences()
        weights = {'PLAY': 0.33, 'DISCARD': 0.33, 'HINT': 0.33}
        
        # Get current risk tolerance
        risk_tolerance = self._calculate_current_risk_tolerance(observation)
        
        # Apply risk tolerance to base weights
        weights['PLAY'] *= risk_tolerance
        weights['DISCARD'] *= (1 - risk_tolerance)
        weights['HINT'] *= (2 - risk_tolerance)  # More hints when being conservative
        
        # Personality-based modifications
        if self.personality_traits['risk_taking']:
            if any(move['action_type'] == 'PLAY' for move in legal_moves):
                weights['PLAY'] += 0.2 * risk_tolerance
                weights['DISCARD'] -= 0.1
                weights['HINT'] -= 0.1
                
        if self.personality_traits['perfectionist']:
            if observation['information_tokens'] > 4:
                weights['HINT'] += 0.2
                weights['PLAY'] -= 0.1 * (1 - risk_tolerance)
                weights['DISCARD'] -= 0.1
                
        if self.personality_traits['methodical']:
            if observation['information_tokens'] < 3:
                weights['DISCARD'] += 0.2
                weights['PLAY'] -= 0.1 * risk_tolerance
                weights['HINT'] -= 0.1
                
        return weights

    def _extract_personality_traits(self):
        """Extract relevant personality traits that influence decision making."""
        persona_lower = str(self.persona).lower()
        traits = {
            'analytical': any(trait in persona_lower for trait in ['analytical', 'logical', 'thoughtful']),
            'social': any(trait in persona_lower for trait in ['helping others', 'empathy', 'friendly']),
            'risk_taking': any(trait in persona_lower for trait in ['confident', 'ambitious', 'outgoing']),
            'perfectionist': any(trait in persona_lower for trait in ['detail-oriented', 'perfectionist', 'meticulous']),
            'intuitive': any(trait in persona_lower for trait in ['intuitive', 'gut feeling', 'instinct']),
            'methodical': any(trait in persona_lower for trait in ['systematic', 'organized', 'structured'])
        }
        return traits

    def _get_hint_preferences(self):
        """Determine hint preferences based on personality."""
        preferences = {
            'color_hint_weight': 0.5,
            'number_hint_weight': 0.5,
            'future_focused': False,
            'efficiency_focused': False
        }
        
        if self.personality_traits['analytical']:
            # Analytical players prefer number hints as they provide more concrete information
            preferences['number_hint_weight'] = 0.7
            preferences['color_hint_weight'] = 0.3
            preferences['efficiency_focused'] = True
            
        elif self.personality_traits['intuitive']:
            # Intuitive players prefer color hints as they give broader information
            preferences['color_hint_weight'] = 0.7
            preferences['number_hint_weight'] = 0.3
            preferences['future_focused'] = True
            
        elif self.personality_traits['methodical']:
            # Methodical players balance hints but focus on efficiency
            preferences['efficiency_focused'] = True
            
        return preferences

    def _evaluate_must_play_rules(self, observation):
        """Evaluate strict rules for when cards must be played."""
        fireworks = observation['fireworks']
        known_cards = observation['card_knowledge'][0]
        
        for card_index, card_knowledge in enumerate(known_cards):
            # Rule 1: If all stacks are at 0 and you know you have a 1, you must play it
            if all(value == 0 for value in fireworks.values()):
                if card_knowledge['rank'] == 0:  # rank 0 = card value 1
                    return {'action_type': 'PLAY', 'card_index': card_index}
            
            # Rule 2: If you know a card completes a stack, you must play it
            if card_knowledge['color'] and card_knowledge['rank'] is not None:
                if fireworks[card_knowledge['color']] == card_knowledge['rank']:
                    return {'action_type': 'PLAY', 'card_index': card_index}
                    
            # Rule 3: If you have perfect information about a playable card, play it
            if card_knowledge['color'] and card_knowledge['rank'] is not None:
                if fireworks[card_knowledge['color']] == card_knowledge['rank']:
                    return {'action_type': 'PLAY', 'card_index': card_index}
        
        return None

    def _evaluate_must_save_rules(self, observation):
        """Evaluate strict rules for when cards must not be discarded."""
        known_cards = observation['card_knowledge'][0]
        
        for card_index, card_knowledge in enumerate(known_cards):
            # Rule 1: Never discard known 5s
            if card_knowledge['rank'] == 4:  # rank 4 = card value 5
                return card_index
            
            # Rule 2: Never discard last copy of a card
            if self._is_last_copy(observation, card_knowledge):
                return card_index
                
        return None

    def _is_last_copy(self, observation, card_knowledge):
        """Check if a card is the last copy based on discard pile and played cards."""
        if not (card_knowledge['color'] and card_knowledge['rank'] is not None):
            return False
            
        color = card_knowledge['color']
        rank = card_knowledge['rank']
        
        # Count copies in discard pile
        discarded = sum(1 for card in observation['discard_pile'] 
                       if card['color'] == color and card['rank'] == rank)
                       
        # Check number of max copies for this rank
        max_copies = 1 if rank == 4 else (2 if rank == 3 else 3)  # 5s: 1 copy, 4s: 2 copies, others: 3 copies
        
        return discarded == max_copies - 1

    def _evaluate_persona_rules(self, observation, legal_moves):
        """Apply personality-based rules for decision making."""
        hint_prefs = self._get_hint_preferences()
        weights = {'PLAY': 0.33, 'DISCARD': 0.33, 'HINT': 0.33}
        
        # Personality-based rule modifications
        if self.personality_traits['risk_taking']:
            # Risk-takers are more likely to play with partial information
            if any(move['action_type'] == 'PLAY' for move in legal_moves):
                weights['PLAY'] += 0.2
                weights['DISCARD'] -= 0.1
                weights['HINT'] -= 0.1
                
        if self.personality_traits['perfectionist']:
            # Perfectionists prefer to hint and get more information before playing
            if observation['information_tokens'] > 4:
                weights['HINT'] += 0.2
                weights['PLAY'] -= 0.1
                weights['DISCARD'] -= 0.1
                
        if self.personality_traits['methodical']:
            # Methodical players prefer to discard when low on information tokens
            if observation['information_tokens'] < 3:
                weights['DISCARD'] += 0.2
                weights['PLAY'] -= 0.1
                weights['HINT'] -= 0.1
                
        return weights

    def _select_hint_move(self, legal_moves, observation):
        """Select the best hint move based on personality and game state."""
        hint_prefs = self._get_hint_preferences()
        color_moves = [m for m in legal_moves if m['action_type'] == 'REVEAL_COLOR']
        number_moves = [m for m in legal_moves if m['action_type'] == 'REVEAL_RANK']
        
        if not color_moves and not number_moves:
            return None
            
        # Apply personality-based hint selection
        if hint_prefs['efficiency_focused']:
            # Prefer hints that reveal information about more cards
            return max(color_moves + number_moves,
                      key=lambda m: len(self._get_affected_cards(m, observation)))
        
        if hint_prefs['future_focused']:
            # Prefer hints about cards that will be playable soon
            return self._select_future_focused_hint(color_moves + number_moves, observation)
            
        # Default to weighted random selection based on preference
        all_moves = []
        weights = []
        
        for move in color_moves:
            all_moves.append(move)
            weights.append(hint_prefs['color_hint_weight'])
            
        for move in number_moves:
            all_moves.append(move)
            weights.append(hint_prefs['number_hint_weight'])
            
        if all_moves:
            weights = np.array(weights) / sum(weights)
            return np.random.choice(all_moves, p=weights)
            
        return None

    def _get_affected_cards(self, hint_move, observation):
        """Get list of cards affected by a hint move."""
        target_hand = observation['observed_hands'][hint_move['target_offset']]
        if hint_move['action_type'] == 'REVEAL_COLOR':
            return [i for i, card in enumerate(target_hand) 
                   if card['color'] == hint_move['color']]
        else:  # REVEAL_RANK
            return [i for i, card in enumerate(target_hand) 
                   if card['rank'] == hint_move['rank']]

    def _select_future_focused_hint(self, hint_moves, observation):
        """Select hint that helps with cards that will be playable soon."""
        fireworks = observation['fireworks']
        best_move = None
        best_score = -1
        
        for move in hint_moves:
            affected_cards = self._get_affected_cards(move, observation)
            target_hand = observation['observed_hands'][move['target_offset']]
            
            score = 0
            for card_idx in affected_cards:
                card = target_hand[card_idx]
                if card['rank'] == fireworks[card['color']]:
                    score += 3  # Immediately playable
                elif card['rank'] == fireworks[card['color']] + 1:
                    score += 2  # Will be playable next
                elif card['rank'] == fireworks[card['color']] + 2:
                    score += 1  # Will be playable soon
                    
            if score > best_score:
                best_score = score
                best_move = move
                
        return best_move or hint_moves[0]
    
    def _explain_hint_choice(self, hint_action, observation):
        """Explain why this specific hint was chosen."""
        target_offset = hint_action['target_offset']
        target_hand = observation['observed_hands'][target_offset]
        affected_cards = self._get_affected_cards(hint_action, observation)
        
        # Calculate the target player's actual name using the offset
        current_player_id = observation['current_player']
        target_player_id = (current_player_id + target_offset) % len(self.all_player_names)
        target_name = self.all_player_names[target_player_id]
        
        explanation = []
        
        # Explain what the hint reveals
        if hint_action['action_type'] == 'REVEAL_COLOR':
            color = hint_action['color']
            explanation.append(f"Revealing {color} to {target_name}")
            explanation.append(f"This hint affects {len(affected_cards)} cards")
            
            # Check if any of these cards are immediately playable
            for card_idx in affected_cards:
                card = target_hand[card_idx]
                if card['rank'] == observation['fireworks'][card['color']]:
                    explanation.append(f"Card {card_idx} is immediately playable")
                    
        else:  # REVEAL_RANK
            rank = hint_action['rank'] + 1  # Convert from 0-based to 1-based
            explanation.append(f"Revealing rank {rank} to {target_name}")
            explanation.append(f"This hint affects {len(affected_cards)} cards")
            
            # Check if any of these cards are immediately playable
            playable_colors = [color for color, height in observation['fireworks'].items() 
                            if height + 1 == rank]
            if playable_colors:
                explanation.append(f"Rank {rank} is currently playable in: {', '.join(playable_colors)}")
        
        # Add personality-based reasoning
        if self.personality_traits['analytical']:
            explanation.append("Chose this hint based on immediate playability")
        elif self.personality_traits['social']:
            explanation.append("Trying to help team coordination")
        elif self.personality_traits['intuitive']:
            explanation.append("This hint should help future plays")
            
        return " | ".join(explanation)

    def _explain_discard_choice(self, discard_action, observation):
        """Explain why this specific card was chosen for discard."""
        card_index = discard_action['card_index']
        card_knowledge = observation['card_knowledge'][0][card_index]
        info_tokens = observation['information_tokens']
        
        explanation = []
        explanation.append(f"Discarding card {card_index}")
        
        # Explain based on known information
        if card_knowledge['color'] or card_knowledge['rank'] is not None:
            known_info = []
            if card_knowledge['color']:
                known_info.append(f"color {card_knowledge['color']}")
            if card_knowledge['rank'] is not None:
                known_info.append(f"rank {card_knowledge['rank'] + 1}")
            explanation.append(f"Known information: {', '.join(known_info)}")
            
            # Check if this card is potentially playable
            if card_knowledge['color'] and card_knowledge['rank'] is not None:
                if observation['fireworks'][card_knowledge['color']] > card_knowledge['rank']:
                    explanation.append("This card is no longer needed")
        else:
            explanation.append("No hints received about this card")
        
        # Explain based on game state
        if info_tokens == 0:
            explanation.append("Must discard to regain information tokens")
        elif info_tokens < 4:
            explanation.append("Discarding to increase available information tokens")
            
        # Add personality-based reasoning
        if self.personality_traits['methodical']:
            explanation.append("Choosing safest card to discard")
        elif self.personality_traits['risk_taking']:
            explanation.append("Taking calculated risk with this discard")
            
        return " | ".join(explanation)

    def _analyze_received_hints(self, observation):
        """Analyze hints received about our cards and determine playability."""
        fireworks = observation['fireworks']
        known_cards = observation['card_knowledge'][0]  # Current player's hand knowledge
        playability_analysis = []
        
        for card_index, card_knowledge in enumerate(known_cards):
            # Start building analysis for this card
            analysis = {
                'card_index': card_index,
                'playability': 'unknown',
                'confidence': 0.0,
                'reasoning': [],
                'known_info': []
            }
            
            # Track what we know about the card
            if card_knowledge['color'] is not None:
                analysis['known_info'].append(f"Known color: {card_knowledge['color']}")
                stack_height = fireworks[card_knowledge['color']]
                analysis['reasoning'].append(f"The {card_knowledge['color']} stack is at {stack_height}")
                
            if card_knowledge['rank'] is not None:
                card_value = card_knowledge['rank'] + 1
                analysis['known_info'].append(f"Known rank: {card_value}")
                
            # If we know nothing about the card, note that explicitly
            if not analysis['known_info']:
                analysis['known_info'].append("No hints received yet")
            
            # Now evaluate playability based on known information
            if card_knowledge['color'] is not None or card_knowledge['rank'] is not None:
                # Case 1: All stacks at 0, got a 1
                if all(v == 0 for v in fireworks.values()) and card_knowledge['rank'] == 0:
                    analysis['playability'] = 'must_play'
                    analysis['confidence'] = 1.0
                    analysis['reasoning'].append("This is a 1 and all stacks are at 0")
                
                # Case 2: Know both color and rank
                elif card_knowledge['color'] and card_knowledge['rank'] is not None:
                    stack_height = fireworks[card_knowledge['color']]
                    card_value = card_knowledge['rank'] + 1
                    
                    if card_value == stack_height + 1:
                        analysis['playability'] = 'must_play'
                        analysis['confidence'] = 1.0
                        analysis['reasoning'].append(
                            f"This is a {card_value} {card_knowledge['color']} and that stack is at {stack_height}")
                    elif card_value <= stack_height:
                        analysis['playability'] = 'never_play'
                        analysis['confidence'] = 1.0
                        analysis['reasoning'].append(
                            f"This {card_value} {card_knowledge['color']} is already played (stack at {stack_height})")
                    elif card_value > stack_height + 1:
                        analysis['playability'] = 'save'
                        analysis['confidence'] = 1.0
                        analysis['reasoning'].append(
                            f"This {card_value} {card_knowledge['color']} needs to wait (stack at {stack_height})")
                
                # Case 3: Only know color
                elif card_knowledge['color']:
                    stack_height = fireworks[card_knowledge['color']]
                    if stack_height == 4:  # Only 5 is playable
                        analysis['playability'] = 'likely_play'
                        analysis['confidence'] = 0.8
                        analysis['reasoning'].append(
                            f"This is {card_knowledge['color']} and that stack needs a 5")
                    else:
                        analysis['playability'] = 'maybe_play'
                        analysis['confidence'] = 0.4
                        analysis['reasoning'].append(
                            f"This is {card_knowledge['color']} and that stack is at {stack_height}")
                
                # Case 4: Only know rank
                elif card_knowledge['rank'] is not None:
                    card_value = card_knowledge['rank'] + 1
                    playable_in = [color for color, height in fireworks.items() 
                                if height + 1 == card_value]
                    if playable_in:
                        analysis['playability'] = 'likely_play'
                        analysis['confidence'] = 0.6
                        analysis['reasoning'].append(
                            f"This is a {card_value} and could be playable in {playable_in}")
                    elif card_value == 1 and any(v == 0 for v in fireworks.values()):
                        analysis['playability'] = 'likely_play'
                        analysis['confidence'] = 0.7
                        analysis['reasoning'].append(
                            f"This is a 1 and some stacks are still at 0")
                    elif all(height >= card_value for height in fireworks.values()):
                        analysis['playability'] = 'never_play'
                        analysis['confidence'] = 1.0
                        analysis['reasoning'].append(
                            f"This {card_value} is no longer needed (all stacks ≥ {card_value})")
                    else:
                        analysis['playability'] = 'maybe_play'
                        analysis['confidence'] = 0.3
                        analysis['reasoning'].append(
                            f"This is a {card_value} but we don't know its color")
                
            playability_analysis.append(analysis)
        
        return playability_analysis

    def _interpret_hint_intention(self, observation, hinter_id):
        """Interpret why another player might have given us specific hints."""
        hint_analysis = self._analyze_received_hints(observation)
        fireworks = observation['fireworks']
        
        # Look at hints in context of other players' known traits
        hinter = [agent for agent in self.game.agents if agent.id == hinter_id][0]
        hinter_traits = hinter.personality_traits
        
        for analysis in hint_analysis:
            # Analytical players tend to hint for immediate plays
            if hinter_traits['analytical'] and analysis['playability'] in ['must_play', 'likely_play']:
                analysis['confidence'] += 0.1
                analysis['reasoning'].append("Hinter is analytical and likely hinting for immediate play")
                
            # Social players might hint to prevent discards
            elif hinter_traits['social'] and analysis['playability'] == 'save':
                analysis['confidence'] += 0.1
                analysis['reasoning'].append("Hinter is social and might be trying to protect this card")
                
            # Risk-taking players might hint about future plays
            elif hinter_traits['risk_taking'] and analysis['playability'] == 'maybe_play':
                analysis['confidence'] += 0.1
                analysis['reasoning'].append("Hinter is risk-taking and might be planning ahead")
        
        return hint_analysis

    def act(self, observation):
        """Enhanced action selection with hint interpretation and risk management."""
        self.turns_played += 1
    
        """Main action selection method incorporating all rules and personality."""
        print(f"\n{self.name}'s turn:")
        legal_moves = observation['legal_moves']
        
        # First analyze any hints we've received
        hint_analysis = self._analyze_received_hints(observation)
        
        # Hint analysis
        print("\nAnalyzing received hints:")
        for analysis in hint_analysis:
            if analysis['known_info']:
                print(f"\nCard {analysis['card_index']}:")
                print("Known information:")
                for info in analysis['known_info']:
                    print(f"  * {info}")
                    
                # Add specific hint intention analysis
                hint_intention = self._analyze_hint_intention(observation, analysis['card_index'])
                print("Hint interpretation:")
                print(f"  * Likely intention: {hint_intention['intention']}")
                print(f"  * Confidence: {hint_intention['confidence']:.2f}")
                for reason in hint_intention['reasoning']:
                    print(f"  * {reason}")
                
                print(f"- Playability: {analysis['playability']}")
                print(f"- Confidence: {analysis['confidence']:.2f}")
                if analysis['reasoning']:
                    print("- Reasoning:")
                    for reason in analysis['reasoning']:
                        print(f"  * {reason}")

        # Adjust play confidence based on hint intentions
        for analysis in hint_analysis:
            if analysis['known_info']:
                hint_intention = self._analyze_hint_intention(observation, analysis['card_index'])
                if hint_intention['intention'] == 'immediate_play':
                    analysis['confidence'] *= 1.5
                elif hint_intention['intention'] == 'save':
                    analysis['confidence'] *= 0.5
        
        # Get current risk tolerance
        risk_tolerance = self._calculate_current_risk_tolerance(observation)
        print(f"\nCurrent risk tolerance: {risk_tolerance:.2f}")
        
        # Regular action selection logic
        must_play = self._evaluate_must_play_rules(observation)
        if must_play and must_play in legal_moves:
            print("\nFollowing must-play rule")
            return must_play
            
        must_save = self._evaluate_must_save_rules(observation)
        if must_save is not None:
            legal_moves = [m for m in legal_moves if 
                        m['action_type'] != 'DISCARD' or 
                        m['card_index'] != must_save]
        
        # Base weights
        weights = self._evaluate_persona_rules(observation, legal_moves)
        
        # Special handling for hints
        hint_moves = [m for m in legal_moves if m['action_type'].startswith('REVEAL')]
        if hint_moves:
            selected_hint = self._select_hint_move(hint_moves, observation)
            if selected_hint:
                weights['HINT'] *= 1.5
        
        # Calculate move weights with safety checks
        move_weights = []
        for move in legal_moves:
            action_type = move['action_type']
            if action_type.startswith('REVEAL'):
                action_type = 'HINT'
            base_weight = weights.get(action_type, 0.33)
            
            # If this is a play move, check if it's one we have hint analysis for
            if action_type == 'PLAY':
                for analysis in hint_analysis:
                    if analysis['card_index'] == move['card_index']:
                        base_weight *= (1 + analysis['confidence'])
            
            # Ensure weight is positive
            base_weight = max(0.01, base_weight)
            move_weights.append(base_weight)
        
        # Normalize weights to ensure they sum to 1
        move_weights = np.array(move_weights)
        move_weights = move_weights / move_weights.sum()
        
        # Verify weights are valid
        if not np.all(move_weights >= 0):
            print("Warning: Invalid weights detected, using uniform distribution")
            move_weights = np.ones(len(legal_moves)) / len(legal_moves)
        
        # Choose action based on weights
        chosen_idx = np.random.choice(len(legal_moves), p=move_weights)
        action = legal_moves[chosen_idx]
        
        # Generate explanation
        print("\nFinal decision reasoning:")
        if action['action_type'] == 'PLAY':
            for analysis in hint_analysis:
                if analysis['card_index'] == action['card_index']:
                    print("Playing based on hint analysis:")
                    for reason in analysis['reasoning']:
                        print(f"- {reason}")
                    print(f"Confidence: {analysis['confidence']:.2f}")
        elif action['action_type'].startswith('REVEAL'):
            print("Giving hint:", self._explain_hint_choice(action, observation))
        elif action['action_type'] == 'DISCARD':
            print("Discarding:", self._explain_discard_choice(action, observation))
        
        print("\nReasoning:")
        if action['action_type'] == 'PLAY':
            print("Decided to play a card:")
            card_knowledge = observation['card_knowledge'][0][action['card_index']]
            if card_knowledge['color'] or card_knowledge['rank'] is not None:
                print("- Based on known information:")
                if card_knowledge['color']:
                    print(f"  * Known color: {card_knowledge['color']}")
                if card_knowledge['rank'] is not None:
                    print(f"  * Known rank: {card_knowledge['rank'] + 1}")
            else:
                print("- Playing with incomplete information")
                if self.personality_traits['risk_taking']:
                    print("- Influenced by risk-taking personality")
        
        return action

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
    """Creates players with random personas from survey data and tracks their names."""
    players = select_and_format_players()
    agents = []
    for i, p in enumerate(players):
        agent = PersonaAgent(
            {"game": game, "information_tokens": 8},
            {"name": p["name"], "persona": p["persona"]}
        )
        agent.id = i  # Add ID to track position
        agent.all_player_names = [p["name"] for p in players]  # Add list of all player names
        agents.append(agent)
    return agents

def run_game():
    """Runs a single game with detailed persona information and move reasoning."""
    environment = rl_env.make('Hanabi-Full', num_players=5)
    observations = environment.reset()
    
    agents = create_players(environment.game)
    done = False
    game_score = 0
    turn_count = 0
    
    # Print initial persona information
    print("\n=== Player Personas ===")
    for agent in agents:
        print(f"\n{agent.name}:")
        print(f"Personality traits: {', '.join(trait for trait, value in agent.personality_traits.items() if value)}")
        print(f"Full description: {agent.persona}")
    print("\n=== Game Start ===\n")
    
    while not done:
        for agent_id, agent in enumerate(agents):
            observation = observations['player_observations'][agent_id]
            
            if observation['current_player'] == agent_id:
                turn_count += 1
                print(f"\n=== Turn {turn_count}: {agent.name}'s Action ===")
                
                # Get game state information
                fireworks = observation['fireworks']
                info_tokens = observation['information_tokens']
                life_tokens = observation['life_tokens']
                
                print(f"Game State:")
                print(f"- Fireworks: {fireworks}")
                print(f"- Information tokens: {info_tokens}")
                print(f"- Life tokens: {life_tokens}")
                
                # Generate the action and capture reasoning
                action = agent.act(observation)
                
                # Print detailed reasoning based on action type
                print("\nReasoning:")
                
                # Check if action was from must-play rules
                must_play = agent._evaluate_must_play_rules(observation)
                if must_play and must_play == action:
                    print("Made a required play based on strict rules:")
                    if all(v == 0 for v in fireworks.values()):
                        print("- All stacks are at 0 and have a known 1")
                    else:
                        print("- Have perfect information about a playable card")
                
                # Explain hint selection reasoning
                elif action['action_type'].startswith('REVEAL'):
                    hint_prefs = agent._get_hint_preferences()
                    print(f"Chose to give a hint ({action['action_type']}):")
                    print(f"- Color hint preference: {hint_prefs['color_hint_weight']:.2f}")
                    print(f"- Number hint preference: {hint_prefs['number_hint_weight']:.2f}")
                    if hint_prefs['future_focused']:
                        print("- Focused on future playability")
                    if hint_prefs['efficiency_focused']:
                        print("- Prioritized hint efficiency")
                
                # Explain discard reasoning
                elif action['action_type'] == 'DISCARD':
                    must_save = agent._evaluate_must_save_rules(observation)
                    if must_save is not None:
                        print("Discarded while protecting critical cards:")
                        print(f"- Protected card index: {must_save}")
                    else:
                        print("Standard discard decision:")
                        print(f"- Information tokens: {info_tokens}/8")
                
                # Explain play reasoning
                elif action['action_type'] == 'PLAY':
                    print("Decided to play a card:")
                    card_knowledge = observation['card_knowledge'][0][action['card_index']]
                    if card_knowledge['color'] or card_knowledge['rank'] is not None:
                        print("- Based on known information:")
                        if card_knowledge['color']:
                            print(f"  * Known color: {card_knowledge['color']}")
                        if card_knowledge['rank'] is not None:
                            print(f"  * Known rank: {card_knowledge['rank'] + 1}")
                    else:
                        print("- Playing with incomplete information")
                        if agent.personality_traits['risk_taking']:
                            print("- Influenced by risk-taking personality")
                
                print(f"\nFinal Action: {action}")
                
                # Execute the action
                observations, reward, done, _ = environment.step(action)
                current_score = sum(observations['player_observations'][0]['fireworks'].values())
                
                if done:
                    game_score = max(0, current_score)
                    break
    
    print(f"\n=== Game Over ===")
    print(f"Final Score: {game_score}")
    return game_score

if __name__ == "__main__":
    score = run_game()