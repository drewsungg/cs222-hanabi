from collections import defaultdict
from openai import OpenAI
from settings import OPENAI_API_KEY, LLM_VERS, DEBUG
import os
import random
from hanabi_learning_environment import rl_env
from hanabi_learning_environment.agents.simple_agent import SimpleAgent
from agents import agent_list  # Import personas
import pandas as pd
import numpy as np
import re

# Create a single client instance to be shared
try:
    client = OpenAI(api_key=OPENAI_API_KEY)
except Exception as e:
    print(f"Warning: Could not initialize OpenAI client: {e}")
    client = None

class HintHistoryEntry:
    """Track details about a given hint"""
    def __init__(self, target_player, card_index, hint_type, value, fireworks_state):
        self.target_player = target_player
        self.card_index = card_index
        self.hint_type = hint_type  # 'color' or 'rank'
        self.value = value  # The color or rank that was hinted
        self.fireworks_state = fireworks_state.copy()  # State of stacks when hint was given
        self.warned = False  # Whether we've warned about this hint becoming invalid

class HanabiLLMAgent(SimpleAgent):
    def __init__(self, config, agent_data):
        super().__init__(config)
        self.name = agent_data["name"]
        self.persona = agent_data["persona"]
        self.all_player_names = []
        self.client = client
        self.personality_traits = self._extract_personality_traits()
        self.hint_history = []  # Track received hints
        self.play_history = []  # Track successful/failed plays
        self.discard_history = []  # Track discarded cards
        self.teammate_play_patterns = {}  # Track teammate play patterns
        self.success_rate = {}  # Track success rate per color/rank
        self.revealed_information = defaultdict(set)  # Track revealed info per player
        self.game_params = config.get("game", {}).get("game_parameters", {})

    def _update_revealed_information(self, move, observation):
        """Update revealed information based on the given hint."""
        if not move['action_type'].startswith('REVEAL'):
            return

        target_player = (observation['current_player'] + move['target_offset']) % len(self.all_player_names)
        target_name = self.all_player_names[target_player]
        
        if move['action_type'] == 'REVEAL_COLOR':
            self.revealed_information[target_name].add(f"color_{move['color']}")
        elif move['action_type'] == 'REVEAL_RANK':
            self.revealed_information[target_name].add(f"rank_{move['rank']}")

    def _is_valid_hint(self, move, observation):
        """
        Check if a hint action is valid, i.e., it affects at least one card
        in the target player's hand.
        """
        if not move['action_type'].startswith('REVEAL'):
            return False
        target_offset = move['target_offset']
        target_player = (observation['current_player'] + target_offset) % len(self.all_player_names)
        target_hand = observation['observed_hands'][target_offset]
        fireworks = observation['fireworks']

        # Check if the hint affects any card
        affects_any = False
        for card in target_hand:
            if card is None:
                continue
            if move['action_type'] == 'REVEAL_COLOR' and card['color'] == move['color']:
                affects_any = True
                break
            if move['action_type'] == 'REVEAL_RANK' and card['rank'] == move['rank']:
                affects_any = True
                break

        if not affects_any:
            return False

        # If it doesn't reveal a playable card or something not already known, degrade.
        # Check if it provides new info:
        target_knowledge = observation['card_knowledge'][target_offset]
        new_info = False
        for i, card in enumerate(target_hand):
            if card is None:
                continue
            c_kn = target_knowledge[i]
            if move['action_type'] == 'REVEAL_COLOR' and card['color'] == move['color'] and c_kn['color'] is None:
                new_info = True
            if move['action_type'] == 'REVEAL_RANK' and card['rank'] == move['rank'] and c_kn['rank'] is None:
                new_info = True
        return new_info

    def _is_hint_redundant(self, move, observation):
        """Check if the hint is redundant based on revealed information."""
        # If it's valid and provides new info, consider it non-redundant.
        return not self._is_valid_hint(move, observation)


    def _track_play_outcome(self, action, observation, success):
        """Track play outcomes to learn from mistakes and successes."""
        card_index = action.get('card_index')
        if card_index is not None:
            card_knowledge = observation['card_knowledge'][0][card_index]
            self.play_history.append({
                'color': card_knowledge.get('color'),
                'rank': card_knowledge.get('rank'),
                'success': success,
                'fireworks_state': observation['fireworks'].copy(),
                'life_tokens': observation['life_tokens']
            })
            
            # Update success rate statistics
            key = f"{card_knowledge.get('color', 'unknown')}_{card_knowledge.get('rank', 'unknown')}"
            if key not in self.success_rate:
                self.success_rate[key] = {'success': 0, 'total': 0}
            self.success_rate[key]['total'] += 1
            if success:
                self.success_rate[key]['success'] += 1

    def _predict_play_success(self, card_knowledge, observation):
        """Predict likelihood of successful play based on historical data."""
        key = f"{card_knowledge.get('color', 'unknown')}_{card_knowledge.get('rank', 'unknown')}"
        if key in self.success_rate:
            stats = self.success_rate[key]
            if stats['total'] > 0:
                return stats['success'] / stats['total']
        return 0.5  # Default 50% if no history

    def _evaluate_teammate_patterns(self, observation):
        """Analyze teammate play patterns to predict their needs."""
        current_player = observation['current_player']
        patterns = {}
        
        for player_offset in range(1, len(observation['observed_hands'])):
            player_id = (current_player + player_offset) % len(observation['observed_hands'])
            if player_id not in self.teammate_play_patterns:
                self.teammate_play_patterns[player_id] = {
                    'successful_colors': defaultdict(int),
                    'successful_ranks': defaultdict(int),
                    'total_plays': 0
                }
            
            patterns[player_id] = {
                'preferred_colors': sorted(
                    self.teammate_play_patterns[player_id]['successful_colors'].items(),
                    key=lambda x: x[1],
                    reverse=True
                ),
                'preferred_ranks': sorted(
                    self.teammate_play_patterns[player_id]['successful_ranks'].items(),
                    key=lambda x: x[1],
                    reverse=True
                )
            }
            
        return patterns

    def _optimize_hint_selection(self, observation, legal_moves):
        """Select the most valuable hint based on teammate patterns and game state."""
        hint_moves = [m for m in legal_moves if m['action_type'].startswith('REVEAL')]
        
        if not hint_moves:
            return None
            
        # Filter out redundant hints
        non_redundant_hints = [
            move for move in hint_moves if not self._is_hint_redundant(move, observation)
        ]

        if not non_redundant_hints:
            return None

        scored_hints = []
        for move in non_redundant_hints:
            value = self._evaluate_hint_value(move, observation)
            scored_hints.append((move, value))

        if not scored_hints:
            return None

        teammate_patterns = self._evaluate_teammate_patterns(observation)
        hint_values = []
        
        for move in legal_moves:
            if not move['action_type'].startswith('REVEAL'):
                continue
                
            target_offset = move['target_offset']
            target_id = (observation['current_player'] + target_offset) % len(observation['observed_hands'])
            target_hand = observation['observed_hands'][target_offset]
            
            value = 0.0
            affected_cards = 0
            playable_cards = 0
            
            for card_idx, card in enumerate(target_hand):
                if card is None:
                    continue
                    
                # Check if hint would affect this card
                affects_card = False
                if move['action_type'] == 'REVEAL_COLOR' and card['color'] == move['color']:
                    affects_card = True
                elif move['action_type'] == 'REVEAL_RANK' and card['rank'] == move['rank']:
                    affects_card = True
                    
                if affects_card:
                    affected_cards += 1
                    if self._is_card_playable(card, observation['fireworks']):
                        playable_cards += 1
                        value += 2.0  # Higher value for playable cards
                        
                    # Add value based on teammate's historical success with this type
                    if target_id in teammate_patterns:
                        if move['action_type'] == 'REVEAL_COLOR':
                            color_prefs = dict(teammate_patterns[target_id]['preferred_colors'])
                            value += color_prefs.get(card['color'], 0) * 0.2
                        else:
                            rank_prefs = dict(teammate_patterns[target_id]['preferred_ranks'])
                            value += rank_prefs.get(card['rank'], 0) * 0.2
            
            # Normalize value by number of affected cards
            if affected_cards > 0:
                value = value / affected_cards
                if playable_cards > 0:
                    value *= 1.5  # Boost value if hint reveals playable cards
                    
            hint_values.append((move, value))
            
        if hint_values:
            return max(hint_values, key=lambda x: x[1])[0]
        return None

    def _is_card_playable(self, card, fireworks):
        """Check if a card is immediately playable."""
        if card['color'] is None or card['rank'] is None:
            return False
        return fireworks[card['color']] == card['rank']
    
    def _analyze_discard_safety(self, observation, card_index):
        """Analyze whether it's safe to discard a card based on game state."""
        card = observation['card_knowledge'][0][card_index]
        fireworks = observation['fireworks']
        discard_pile = observation['discard_pile']

        card_knowledge = observation['card_knowledge'][0][card_index]
        
        # If the card has hints, it's not safe to discard
        if card_knowledge['color'] is not None or card_knowledge['rank'] is not None:
            return 0.0  # Hinted cards should not be discarded
        
        if card['color'] is None and card['rank'] is None:
            return 0.7  # Moderately safe to discard unknown cards
            
        risk_score = 0.0
        
        # Check if card is already played in fireworks
        if card['color'] and card['rank'] is not None:
            if fireworks[card['color']] > card['rank']:
                return 1.0  # Safe to discard if already played
                
        # Check if it's the last copy
        if self._is_last_copy(observation, card):
            return 0.0  # Never discard last copy
            
        # Check if card is potentially needed soon
        if card['color']:
            current_height = fireworks[card['color']]
            if card['rank'] is not None and card['rank'] == current_height:
                return 0.1  # Very risky to discard next needed card
            
        return 0.5  # Moderate risk for other cases
    
    def get_player_name(self, current_player, target_offset):
        """Get the name of a player based on current player and target offset."""
        if not self.all_player_names:
            return f"Player {target_offset}"  # Fallback if names aren't set
        target_player_id = (current_player + target_offset) % len(self.all_player_names)
        return self.all_player_names[target_player_id]
    
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

    def _evaluate_hint_value(self, move, legal_moves, observation):
        """Evaluate hint value without recursion."""
        if not move['action_type'].startswith('REVEAL'):
            return 0.0

        target_offset = move['target_offset']
        target_hand = observation['observed_hands'][target_offset]
        fireworks = observation['fireworks']
        value = 0.0
        new_information = False

        # Check each affected card
        for card_idx, card in enumerate(target_hand):
            if card is None:
                continue

            # Determine if the hint provides useful information
            if move['action_type'] == 'REVEAL_COLOR' and card['color'] == move['color']:
                if observation['card_knowledge'][target_offset][card_idx]['color'] is None:
                    new_information = True
                # Value the card based on playability or importance
                if fireworks[card['color']] == card['rank']:
                    value += 2.0  # High value for immediately playable cards
                elif fireworks[card['color']] == card['rank'] - 1:
                    value += 1.0  # Useful setup for future play

            elif move['action_type'] == 'REVEAL_RANK' and card['rank'] == move['rank']:
                if observation['card_knowledge'][target_offset][card_idx]['rank'] is None:
                    new_information = True
                if fireworks[card['color']] == card['rank']:
                    value += 2.0
                elif fireworks[card['color']] == card['rank'] - 1:
                    value += 1.0

        # Factor in new information
        if new_information:
            value *= 1.5  # Boost value for providing new information

        return value

    def _analyze_hint_intention(self, hint_move, observation):
        """
        Explain the strategic intention behind giving a specific hint.
        Used for logging and understanding the AI's decision making.
        """
        target_name = self.get_player_name(observation['current_player'], hint_move['target_offset'])
        hint_type = "color" if hint_move['action_type'] == 'REVEAL_COLOR' else "rank"
        hint_value = hint_move['color'] if hint_type == 'color' else hint_move['rank'] + 1
        
        # Get the last hint analysis if available
        analysis = getattr(self, 'last_hint_analysis', None)
        if not analysis:
            return f"Telling {target_name} about {hint_type} {hint_value}"
            
        # Construct detailed explanation
        explanation = f"Telling {target_name} about {hint_type} {hint_value} "
        explanation += f"(affecting {analysis['affected_cards']} cards) because:\n"
        
        # Add each strategic purpose
        for purpose in analysis['purposes']:
            explanation += f"- {purpose}\n"
            
        return explanation.strip()

    def _interpret_recent_hints(self, observation):
        """Analyze the sequence of hints received to understand team strategy."""
        all_hints_analysis = []
        fireworks = observation['fireworks']
        
        # Look at cards that have hints
        for card_idx, card_knowledge in enumerate(observation['card_knowledge'][0]):
            if card_knowledge['color'] is not None or card_knowledge['rank'] is not None:
                analysis = {
                    'card_index': card_idx,
                    'intention': 'unknown',
                    'confidence': 0.0,
                    'reasoning': []
                }

                # If we know color
                if card_knowledge['color'] is not None:
                    color = card_knowledge['color']
                    if fireworks[color] == 0:
                        analysis['intention'] = 'potential_play'
                        analysis['confidence'] = 0.8
                        analysis['reasoning'].append(f"This {color} card might be a 1 to start the stack")
                    elif fireworks[color] == 4:
                        analysis['intention'] = 'potential_play'
                        analysis['confidence'] = 0.8
                        analysis['reasoning'].append(f"This might be the {color} 5 needed to complete the stack")
                    else:
                        next_value = fireworks[color] + 1
                        analysis['intention'] = 'information'
                        analysis['confidence'] = 0.5
                        analysis['reasoning'].append(f"This {color} card might be the {next_value} needed next")

                # If we know rank
                if card_knowledge['rank'] is not None:
                    value = card_knowledge['rank'] + 1
                    if value == 1 and any(h == 0 for h in fireworks.values()):
                        analysis['intention'] = 'potential_play'
                        analysis['confidence'] = 0.9
                        analysis['reasoning'].append(f"This is a 1 and could start any empty stack")
                    elif value == 5:
                        analysis['intention'] = 'save'
                        analysis['confidence'] = 0.8
                        analysis['reasoning'].append("This is a 5 and should be saved")
                    else:
                        playable_colors = [color for color, height in fireworks.items() if height == value - 1]
                        if playable_colors:
                            analysis['intention'] = 'potential_play'
                            analysis['confidence'] = 0.7
                            analysis['reasoning'].append(f"This {value} could be playable in these colors: {', '.join(playable_colors)}")

                # If we know both
                if card_knowledge['color'] is not None and card_knowledge['rank'] is not None:
                    value = card_knowledge['rank'] + 1
                    color = card_knowledge['color']
                    if fireworks[color] == card_knowledge['rank']:
                        analysis['intention'] = 'immediate_play'
                        analysis['confidence'] = 1.0
                        analysis['reasoning'] = [f"This {color} {value} is immediately playable"]
                
                # Adjust confidence based on game state
                if observation['life_tokens'] == 1:
                    analysis['confidence'] *= 0.8
                    analysis['reasoning'].append("Being more cautious due to one life remaining")
                if observation['information_tokens'] <= 2:
                    analysis['confidence'] *= 1.2
                    analysis['reasoning'].append("Hint was given despite low information tokens, likely important")

                all_hints_analysis.append(analysis)
        
        # Sort analyses by confidence
        all_hints_analysis.sort(key=lambda x: x['confidence'], reverse=True)
        
        return all_hints_analysis

    def _optimize_hint_selection(self, observation, legal_moves):
        """Select best hint based on strategic value with validation."""
        hint_moves = [m for m in legal_moves if m['action_type'].startswith('REVEAL')]
        if not hint_moves:
            return None
            
        # Filter out invalid hints first
        valid_hints = []
        for move in hint_moves:
            target_hand = observation['observed_hands'][move['target_offset']]
            
            # Check if hint would affect any cards
            if move['action_type'] == 'REVEAL_COLOR':
                affects_cards = any(
                    card['color'] == move['color'] 
                    for card in target_hand 
                    if card is not None
                )
            else:  # REVEAL_RANK
                affects_cards = any(
                    card['rank'] == move['rank'] 
                    for card in target_hand 
                    if card is not None
                )
                
            if affects_cards:
                valid_hints.append(move)
                
        # Score only valid hints
        scored_hints = []
        for move in valid_hints:
            value = self._evaluate_hint_value(move, legal_moves, observation)
            scored_hints.append((move, value))
            
        if not scored_hints:
            return None
            
        return max(scored_hints, key=lambda x: x[1])[0]

    def _evaluate_hint_based_play(self, observation, legal_moves):
        """Enhance hint-driven logic to prioritize hinted cards."""
        hint_analyses = self._interpret_recent_hints(observation)
        playable_cards = []

        for analysis in hint_analyses:
            if analysis['intention'] in ['immediate_play', 'potential_play']:
                card_index = analysis['card_index']
                play_moves = [m for m in legal_moves if m['action_type'] == 'PLAY' and m['card_index'] == card_index]
                if play_moves:
                    playable_cards.append({
                        'move': play_moves[0],
                        'confidence': analysis['confidence'],
                        'reasoning': analysis['reasoning']
                    })

        if playable_cards:
            playable_cards.sort(key=lambda x: x['confidence'], reverse=True)
            return playable_cards[0]

        return None

    def _evaluate_strict_rules(self, observation, legal_moves):
        """Evaluate strict must-follow rules that override other logic."""
        fireworks = observation['fireworks']
        known_cards = observation['card_knowledge'][0]

        # Rule 1: Play hinted cards if they are definitely playable
        for move in legal_moves:
            if move['action_type'] == 'PLAY':
                card = known_cards[move['card_index']]
                if card['color'] is not None and card['rank'] is not None:
                    if fireworks[card['color']] == card['rank']:
                        return move, "Playing hinted card that is confirmed playable"

        # Rule 2: Avoid discarding hinted cards
        for move in legal_moves:
            if move['action_type'] == 'DISCARD':
                card_index = move['card_index']
                card_knowledge = known_cards[card_index]
                if card_knowledge['color'] is not None or card_knowledge['rank'] is not None:
                    continue  # Skip discards for hinted cards
                # Allow discarding only if absolutely necessary (e.g., no other moves)
                return None, "Avoiding discard of hinted cards"

        # Rule 3: If no safe discard is available, fall back to hints or other moves
        return None, None
    
    def _evaluate_protective_rules(self, observation):
        """Identify cards that must be protected from discard."""
        fireworks = observation['fireworks']
        known_cards = observation['card_knowledge'][0]
        protected_cards = set()
        
        for card_index, card in enumerate(known_cards):
            # Protect known 5s
            if card['rank'] == 4:  # rank 4 = value 5
                protected_cards.add(card_index)
                
            # Protect last copy of a playable card
            if card['color'] and card['rank'] is not None:
                if self._is_last_copy(observation, card) and card['rank'] >= fireworks[card['color']]:
                    protected_cards.add(card_index)
                    
        return protected_cards

    def _analyze_hint_value(self, move, observation):
        """Analyze the strategic value of a hint."""
        if not move['action_type'].startswith('REVEAL'):
            return 0.0
            
        target_offset = move['target_offset']
        target_hand = observation['observed_hands'][target_offset]
        fireworks = observation['fireworks']
        value = 0.0
        
        # Check each affected card
        for card_idx, card in enumerate(target_hand):
            if card is None:
                continue
                
            if move['action_type'] == 'REVEAL_COLOR' and card['color'] == move['color']:
                if fireworks[card['color']] == card['rank']:
                    value += 1.0  # Immediately playable
                elif fireworks[card['color']] == card['rank'] - 1:
                    value += 0.5  # Will be playable next
                    
            elif move['action_type'] == 'REVEAL_RANK' and card['rank'] == move['rank']:
                if any(fireworks[color] == card['rank'] for color in fireworks):
                    value += 1.0  # Immediately playable
                elif any(fireworks[color] == card['rank'] - 1 for color in fireworks):
                    value += 0.5  # Will be playable next
                    
        return value

    def _get_personality_biases(self):
        """Get base action biases based on personality traits."""
        biases = {
            'PLAY': 0.0,
            'DISCARD': 0.0,
            'HINT': 0.0
        }
        
        # Adjust based on personality traits
        if self.personality_traits['risk_taking']:
            biases['PLAY'] += 0.4
        if self.personality_traits['analytical']:
            biases['DISCARD'] += 0.05
        if self.personality_traits['social']:
            biases['HINT'] += 0.08
            
        return biases

    def _is_last_copy(self, observation, card):
        """Check if a card is the last copy available."""
        if not card['color'] or card['rank'] is None:
            return False
            
        # Count copies in discard pile
        discarded = 0
        for discard in observation['discard_pile']:
            if discard['color'] == card['color'] and discard['rank'] == card['rank']:
                discarded += 1
                
        # Get max copies based on card rank
        if card['rank'] == 4:
            max_copies = 1
        elif card['rank'] == 1 or card['rank'] == 2 or card['rank'] == 3:
            max_copies = 2
        else:
            max_copies = 3
            
        return discarded >= max_copies - 1

    def _format_known_cards(self, observation):
        """Format known card information."""
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
        """Format legal moves for clear display."""
        formatted = []
        for i, move in enumerate(legal_moves):
            if move['action_type'] == 'PLAY':
                formatted.append(f"{i}: Play card {move['card_index']}")
            elif move['action_type'] == 'DISCARD':
                formatted.append(f"{i}: Discard card {move['card_index']}")
            elif move['action_type'].startswith('REVEAL'):
                target_name = self.get_player_name(observation['current_player'], move['target_offset'])
                info_type = 'color' if move['action_type'] == 'REVEAL_COLOR' else 'rank'
                info_value = move['color'] if info_type == 'color' else move['rank'] + 1
                formatted.append(f"{i}: Tell {target_name} about {info_type} {info_value}")
        return "\n".join(formatted)

    def _check_for_dangerous_hints(self, observation):
        """Check if any previously given hints have become dangerous."""
        if not hasattr(self, 'hint_history'):
            self.hint_history = []
            
        fireworks = observation['fireworks']
        urgent_warnings = []
        
        for hint in self.hint_history:
            if hint.warned:
                continue
                
            # Check if this hint has become dangerous
            dangerous = False
            old_state = hint.fireworks_state
            
            # A hint becomes dangerous if:
            # 1. It was about a rank 1 and that stack is now higher
            # 2. Similar for other ranks - any card that's now below the stack height
            if hint.hint_type == 'rank':
                rank_value = hint.value
                # Check each color's stack - if any stack is higher than this rank now,
                # and wasn't when we gave the hint, it's dangerous
                for color, height in fireworks.items():
                    old_height = old_state[color]
                    if height > rank_value and old_height <= rank_value:
                        dangerous = True
                        break
                        
            # For color hints, check if the hinted cards are now unplayable
            elif hint.hint_type == 'color':
                color = hint.value
                if fireworks[color] > old_state[color]:
                    # The stack has grown - some previously playable cards might not be now
                    dangerous = True
                    
            if dangerous and not hint.warned:
                urgent_warnings.append(hint)
                
        return urgent_warnings[0] if urgent_warnings else None

    def _is_warning_hint(self, move, danger_hint, observation):
        """Check if this hint would warn about a dangerous card."""
        if not move['action_type'].startswith('REVEAL'):
            return False
            
        target_offset = move['target_offset']
        # Check if this hint targets the same player and card as the dangerous hint
        target_player = (observation['current_player'] + target_offset) % len(self.all_player_names)
        
        if target_player != danger_hint.target_player:
            return False
            
        # Check if this hint would affect the dangerous card
        target_hand = observation['observed_hands'][target_offset]
        dangerous_card = target_hand[danger_hint.card_index]
        
        if move['action_type'] == 'REVEAL_COLOR':
            return dangerous_card['color'] == move['color']
        else:  # REVEAL_RANK
            return dangerous_card['rank'] == move['rank']
    
    def _get_prompt_template(self):
        return '''As {name}, a Hanabi player who {persona}, I'm analyzing my turn.

    {context}

    Let me think through this step by step:
    1. What's the most critical aspect of the current game state?
    2. Which opportunities or threats need immediate attention?
    3. How can I best support my teammates based on their patterns?
    4. What's the optimal use of our information tokens?
    5. Which move best balances risk vs. reward?
    6. How does this align with my personality and play style?

    After analysis, provide:
    MOVE: [index number]
    REASONING: [clear strategic explanation]
    CONFIDENCE: [0-1 scale]'''

    def _create_prompt(self, context_parts):
        template = self._get_prompt_template()
        context = '\n'.join(context_parts)
        return template.format(
            name=self.name,
            persona=self.persona,
            context=context
        )

    def act(self, observation):
        """
        Execute the best move for the current turn using all helper functions.
        Returns the chosen move.
        """
        legal_moves = observation['legal_moves']
        fireworks = observation['fireworks']
        life_tokens = observation['life_tokens']
        info_tokens = observation['information_tokens']

        print(f"\n{self.name}'s turn:")
        print(f"Persona: {self.persona}")

        # Step 1: Analyze game state
        game_analysis = self._analyze_game_state(observation)
        print("\nGame Analysis:")
        print(game_analysis)

        # Step 2: Evaluate strict rules
        strict_move, strict_reason = self._evaluate_strict_rules(observation, legal_moves)
        if strict_move:
            print(f"\nStrict Rule Applied: {strict_reason}")
            return strict_move

        # Step 3: Evaluate all potential moves
        move_scores = []

        # Evaluate plays
        for move in [m for m in legal_moves if m['action_type'] == 'PLAY']:
            card_index = move['card_index']
            predicted_success = self._predict_play_success(observation['card_knowledge'][0][card_index], observation)
            confidence = predicted_success * self._calculate_current_risk_tolerance(observation)
            move_scores.append((move, confidence, ["Based on play success and risk tolerance"]))

        # Evaluate discards
        for move in [m for m in legal_moves if m['action_type'] == 'DISCARD']:
            card_index = move['card_index']
            safety_score = self._analyze_discard_safety(observation, card_index)
            move_scores.append((move, safety_score, ["Discard safety analysis"]))

        # Evaluate hints
        for move in [m for m in legal_moves if m['action_type'].startswith('REVEAL')]:
            hint_value = self._evaluate_hint_value(move, legal_moves, observation)
            move_scores.append((move, hint_value, ["Hint value based on teammate patterns and card states"]))

        # Prepare LLM input context
        context_parts = []
        context_parts.append(f"Game Analysis: {game_analysis} Fireworks: {fireworks}, Info tokens: {info_tokens}, Life tokens: {life_tokens}")
        context_parts.append(f"My cards: {self._format_known_cards(observation)}")
        context_parts.append(f"Available Moves: {self._format_legal_moves(legal_moves, observation)}")
        context_parts.append(f"Move scores: {move_scores}")
        context_parts.append("\nOutput Format:")
        context_parts.append("""
        Based on the information given and your persona, choose what you think is the most optimal move for hanabi. 
        Your response must follow this strict format:
        **MOVE**: <index> (short description of the move)
        **CONFIDENCE**: <value between 0 and 1>

        For example:
        **MOVE**: 3 (Tell Player A about color Y)
        **CONFIDENCE**: 0.85
        """)
        prompt = self._create_prompt(context_parts)

        try:
            # Get LLM response
            response = self.client.chat.completions.create(
                model=LLM_VERS,
                messages=[
                    {"role": "system", "content": "You are a strategic Hanabi player focused on teamwork and reasoning."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=3000,
                temperature=0.4
            )
            
            content = response.choices[0].message.content
            print("\nLLM Decision Process:")
            print(content)

            move_index, confidence = None, None

            def extract_move_confidence(content):
                move_match = re.search(r"\*\*MOVE\*\*:\s*(\d+)", content)
                confidence_match = re.search(r"\*\*CONFIDENCE\*\*:\s*([\d.]+)", content)

                move_index = int(move_match.group(1)) if move_match else None
                confidence = float(confidence_match.group(1)) if confidence_match else None

                return move_index, confidence

            move_index, confidence = extract_move_confidence(content)

            # Validate the extracted MOVE index
            if move_index is not None and 0 <= move_index < len(legal_moves):
                chosen_move = legal_moves[move_index]

                # Check validity for hint actions
                if chosen_move['action_type'].startswith('REVEAL') and not self._is_valid_hint(chosen_move, observation):
                    raise ValueError(f"Invalid hint: Hint {chosen_move} does not affect any cards in the target's hand.")

                print(f"Chosen MOVE: {move_index}, CONFIDENCE: {confidence:.2f}")
                return chosen_move
            else:
                raise ValueError(f"Invalid MOVE index extracted: {move_index}")

        except Exception as e:
            print(f"LLM error: {e}")

        #     # Fallback: Default to hinting as a safe action if available
        #     hint_moves = [m for m in legal_moves if m['action_type'].startswith('REVEAL') and self._is_valid_hint(m, observation)]
        #     if hint_moves:
        #         print("Fallback MOVE: Hint action chosen as default.")
        #         return random.choice(hint_moves)

        #     # If no valid hint moves, fallback to a discard or play action
        #     print("Fallback MOVE: First legal move chosen as default.")
        #     return legal_moves[0]
            if not move_scores:  # If no moves are scored, fallback to discard
                print("No scored moves available. Falling back to discarding the oldest card.")
                for move in legal_moves:
                    if move['action_type'] == 'DISCARD' and move['card_index'] == 0:
                        print(f"Fallback MOVE: Discard card 0, CONFIDENCE: 0.00")
                        return move

        # Step 5: Choose the optimal move
        best_move, best_score, best_reasons = max(move_scores, key=lambda x: x[1])

        # Log the decision-making process
        print("\nMove Evaluation Results:")
        for move, score, reasons in sorted(move_scores, key=lambda x: x[1], reverse=True)[:3]:
            print(f"{self._format_move(move)} (Score: {score:.2f}):")
            for reason in reasons:
                print(f"- {reason}")

        print(f"\nChosen Move: {self._format_move(best_move)} with score {best_score:.2f}")
        return best_move

    # def act(self, observation):
    #     """
    #     Execute turn with LLM-exclusive decision making.
    #     Returns the chosen move.
    #     """
    #     legal_moves = observation['legal_moves']
    #     fireworks = observation['fireworks']
    #     life_tokens = observation['life_tokens']
    #     info_tokens = observation['information_tokens']

    #     print(f"\n{self.name}'s turn:")
    #     print(f"Persona: {self.persona}")

    #     # Prepare LLM input context
    #     context_parts = []
    #     context_parts.append(f"Game Analysis: Fireworks: {fireworks}, Info tokens: {info_tokens}, Life tokens: {life_tokens}")
    #     context_parts.append(f"My cards: {self._format_known_cards(observation)}")
    #     context_parts.append(f"Available Moves: {self._format_legal_moves(legal_moves, observation)}")
    #     context_parts.append("\nOutput Format:")
    #     context_parts.append("""
    #     Your response must follow this strict format:
    #     **MOVE**: <index> (short description of the move)
    #     **CONFIDENCE**: <value between 0 and 1>

    #     For example:
    #     **MOVE**: 3 (Tell Player A about color Y)
    #     **CONFIDENCE**: 0.85
    #     """)
    #     prompt = self._create_prompt(context_parts)

    #     try:
    #         # Get LLM response
    #         response = self.client.chat.completions.create(
    #             model=LLM_VERS,
    #             messages=[
    #                 {"role": "system", "content": "You are a strategic Hanabi player focused on teamwork and reasoning."},
    #                 {"role": "user", "content": prompt}
    #             ],
    #             max_tokens=3000,
    #             temperature=0.4
    #         )
            
    #         content = response.choices[0].message.content
    #         print("\nLLM Decision Process:")
    #         print(content)

    #         move_index, confidence = None, None
    #         def extract_move_confidence(content):
    #             move_match = re.search(r"\*\*MOVE\*\*:\s*(\d+)", content)
    #             confidence_match = re.search(r"\*\*CONFIDENCE\*\*:\s*([\d.]+)", content)

    #             move_index = int(move_match.group(1)) if move_match else None
    #             confidence = float(confidence_match.group(1)) if confidence_match else None

    #             return move_index, confidence
    
    #         move_index, confidence = extract_move_confidence(content)

    #         # Validate extracted MOVE and CONFIDENCE
    #         if move_index is not None and confidence is not None and 0 <= move_index < len(legal_moves):
    #             print(f"Chosen MOVE: {move_index}, CONFIDENCE: {confidence:.2f}")
    #             return legal_moves[move_index]
    #         else:
    #             print(f"Failed to validate MOVE and CONFIDENCE. MOVE: {move_index}, CONFIDENCE: {confidence}")

    #     except Exception as e:
    #         print(f"LLM error: {e}")

    #     # Fallback MOVE if LLM fails
    #     print("LLM decision-making failed. Defaulting to a safe move.")
    #     for move in legal_moves:
    #         if move['action_type'] == 'DISCARD':  # Default to discard to regain tokens
    #             print(f"Fallback MOVE: {legal_moves.index(move)}, CONFIDENCE: 0.00")
    #             return move
    #     print(f"Fallback MOVE: 0, CONFIDENCE: 0.00")
    #     return legal_moves[0]  # Default to the first legal move as a last resort
        
    def _score_moves(self, observation, legal_moves):
        """Comprehensive move scoring system using all helper functions."""
        move_scores = []
        
        for move in legal_moves:
            score = 0.3  # Base score
            reasons = []
            
            if move['action_type'] == 'PLAY':
                # Use hint analysis
                hint_based_play = self._evaluate_hint_based_play(observation, legal_moves)
                if hint_based_play and move == hint_based_play['move']:
                    score = hint_based_play['confidence'] * 2.0
                    reasons.extend(hint_based_play['reasoning'])
                
                # Incorporate play history and risk tolerance
                card = observation['card_knowledge'][0][move['card_index']]
                predicted_success = self._predict_play_success(card, observation)
                risk_tolerance = self._calculate_current_risk_tolerance(observation)
                score *= predicted_success * risk_tolerance
                reasons.append(f"Historical success rate: {predicted_success:.2f}")
                reasons.append(f"Risk adjusted: {risk_tolerance:.2f}")
                
            elif move['action_type'] == 'DISCARD':
                card_idx = move['card_index']
                protected_cards = self._evaluate_protective_rules(observation)
                
                if card_idx in protected_cards:
                    score = 0.1
                    reasons.append("Protected card")
                else:
                    safety = self._analyze_discard_safety(observation, card_idx)
                    score = safety
                    reasons.append(f"Discard safety: {safety:.2f}")

                    # Add priority for the oldest card
                    if card_idx == 0:  # Assuming oldest card is at index 0
                        score += 0.2
                        reasons.append("Oldest card in hand, preferred for discard")
                
                # Consider info token state
                info_tokens = observation['information_tokens']
                if info_tokens == 0:
                    score *= 2.0
                    reasons.append("Need info tokens")
                elif info_tokens >= 7:
                    score *= 0.5
                    reasons.append("Info tokens plentiful")
                    
            elif move['action_type'].startswith('REVEAL'):
                # Evaluate hint value
                hint_value = self._evaluate_hint_value(move, observation)
                score = hint_value
                reasons.append(f"Hint value: {hint_value:.2f}")
                
                # Consider teammate patterns
                patterns = self._evaluate_teammate_patterns(observation)
                target_id = (observation['current_player'] + move['target_offset']) % len(self.all_player_names)
                if target_id in patterns:
                    success_bonus = 0.1 * len(patterns[target_id]['preferred_colors'])
                    score *= (1 + success_bonus)
                    reasons.append(f"Teammate pattern bonus: {success_bonus:.2f}")
            
            # Apply personality biases
            action_type = 'HINT' if move['action_type'].startswith('REVEAL') else move['action_type']
            bias = self._get_personality_biases().get(action_type, 0)
            score *= (1 + bias)
            if bias != 0:
                reasons.append(f"Personality bias: {bias:+.2f}")
                
            move_scores.append((move, score, reasons))
        
        # Choose and report best move
        best_move, best_score, best_reasons = max(move_scores, key=lambda x: x[1])
        
        print("\nMove Evaluation Results:")
        sorted_moves = sorted(move_scores, key=lambda x: x[1], reverse=True)[:3]
        for move, score, reasons in sorted_moves:
            print(f"\n{self._format_move(move)} (score: {score:.2f}):")
            for reason in reasons:
                print(f"- {reason}")
                
        print(f"\nChosen action: {self._format_move(best_move)}")
        return best_move

    def _format_move(self, move):
        """Format a move for clear output."""
        if move['action_type'] == 'PLAY':
            return f"Play card {move['card_index']}"
        elif move['action_type'] == 'DISCARD':
            return f"Discard card {move['card_index']}"
        elif move['action_type'].startswith('REVEAL'):
            target = self.get_player_name(move.get('current_player', 0), move['target_offset'])
            info_type = 'color' if move['action_type'] == 'REVEAL_COLOR' else 'rank'
            info_value = move['color'] if info_type == 'color' else move['rank'] + 1
            return f"Hint {target} about {info_type} {info_value}"
        return str(move)

    def _analyze_game_state(self, observation):
        """Generate strategic analysis of game state."""
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
        
        # Hint analysis
        hint_analysis = self._interpret_recent_hints(observation)
        if hint_analysis:
            most_confident = max(hint_analysis, key=lambda x: x['confidence'])
            if most_confident['confidence'] > 0.8:
                analysis.append(f"Have high-confidence information about card {most_confident['card_index']}: {most_confident['intention']}")
        
        return " ".join(analysis)
    
    def _calculate_current_risk_tolerance(self, observation):
        """
        Calculate current risk tolerance based on game state.
        Returns a value between 0.0 (extremely cautious) and 5.0 (fully aggressive).
        """
        life_tokens = observation['life_tokens']
        info_tokens = observation['information_tokens']
        fireworks = observation['fireworks']
        
        # Start with baseline risk tolerance
        base_risk = 0.9
        
        # Factor 1: Life Tokens
        # We get more conservative as we lose lives
        if life_tokens == 1:
            base_risk *= 0.4  # Very conservative with last life
        elif life_tokens == 2:
            base_risk *= 0.8  # Moderately conservative with 2 lives
        else:
            base_risk *= 1.0  # Normal risk with 3 lives
            
        # Factor 2: Game Progress
        # Calculate completion percentage (max score is 25)
        total_score = sum(fireworks.values())
        progress = total_score / 25.0
        
        if progress < 0.4:  # Early game
            # More aggressive early to establish stacks
            base_risk *= 2.0
        elif progress > 0.8:  # Late game
            # More conservative late to protect high score
            base_risk *= 0.8
            
        # Factor 3: Information Tokens
        # More conservative when low on info tokens as we can't help teammates
        if info_tokens <= 2:
            base_risk *= 0.8
        elif info_tokens >= 7:
            # Slightly more aggressive with lots of info tokens
            base_risk *= 1.1
            
        # Factor 4: Personality Traits
        if self.personality_traits['risk_taking']:
            base_risk *= 1.2
        if self.personality_traits['perfectionist']:
            base_risk *= 0.8
            
        # Check for critical game states
        critical_conditions = []
        
        # Critical condition 1: Last life token
        if life_tokens == 1:
            critical_conditions.append("last life token")
            
        # Critical condition 2: No information tokens
        if info_tokens == 0:
            critical_conditions.append("no information tokens")
            
        # Critical condition 3: High score at risk
        if total_score > 20:
            critical_conditions.append("high score")
            
        # If any critical conditions exist, cap maximum risk
        if critical_conditions:
            base_risk = min(base_risk, 0.5)
            
        # Ensure final risk tolerance is between 0.1 and 1.0
        return max(0.1, min(5.0, base_risk))

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
    num_players = len(agents)
    
    # Format fireworks (stacks)
    stack_status = [f"{color}: {height}" for color, height in fireworks.items()]

    # Format hand information for all players
    hand_info = []
    for relative_idx, hand in enumerate(observation['observed_hands']):
        # Convert relative index to global player ID
        global_player_id = (current_player_id + relative_idx) % num_players
        player_name = agents[global_player_id].name

        cards_info = []
        # If this is the current player
        if global_player_id == current_player_id:
            # Current player's cards are unknown to themselves, just show hints
            # card_knowledge[0] refers to the current player's hand knowledge
            for card_idx, card_kn in enumerate(observation['card_knowledge'][0]):
                hints = []
                if card_kn['color'] is not None:
                    hints.append(f"color:{card_kn['color']}")
                if card_kn['rank'] is not None:
                    hints.append(f"rank:{card_kn['rank']+1}")
                hint_str = f"({', '.join(hints)})" if hints else "(no hints)"
                cards_info.append(f"Card {card_idx}: {hint_str}")
        else:
            # For other players, you can see their cards fully from current player's perspective
            # observation['card_knowledge'][relative_idx] gives the knowledge about this player’s hand
            card_knowledge_for_player = observation['card_knowledge'][relative_idx]
            for card_idx, card in enumerate(hand):
                if card is None:
                    # Card already played or replaced, just show empty
                    cards_info.append(f"Card {card_idx}: (empty)")
                    continue

                # card is known, we can show its color/rank
                visible_str = f"[{card['color']} {card['rank']+1}]"
                hints = []
                c_kn = card_knowledge_for_player[card_idx]
                if c_kn['color'] is not None:
                    hints.append(f"color:{c_kn['color']}")
                if c_kn['rank'] is not None:
                    hints.append(f"rank:{c_kn['rank']+1}")
                hint_str = f"({', '.join(hints)})" if hints else "(no hints)"
                cards_info.append(f"Card {card_idx}: {visible_str} {hint_str}")

        hand_info.append(f"{player_name}: {' | '.join(cards_info)}")

    state_str = "Current Game State:\n"
    state_str += f"Stacks: {' | '.join(stack_status)}\n"
    state_str += f"Info tokens: {info_tokens} | Life tokens: {life_tokens}\n"
    state_str += "Hands:\n" + '\n'.join(hand_info)

    return state_str

def run_game():
    """
    Runs a single game of Hanabi with enhanced state tracking, move analysis, and detailed logging.
    Returns the final game score.
    """
    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    log_filename = f'hanabi_game_{timestamp}.txt'
    
    with open(log_filename, 'w') as log_file:
        def log_and_print(message):
            print(message)
            log_file.write(message + '\n')
            log_file.flush()
            
        # Initialize game environment and agents
        environment = rl_env.make('Hanabi-Full', num_players=5)
        observations = environment.reset()
        agents = create_players(environment.game)
        
        # Set up player names and initialize game statistics
        for i, agent in enumerate(agents):
            agent.all_player_names = [a.name for a in agents]
            
        # Track game statistics
        game_stats = {
            'turns': 0,
            'hints_given': 0,
            'cards_played': 0,
            'cards_discarded': 0,
            'lives_lost': 0,
            'score_progression': [],
            'player_actions': defaultdict(lambda: defaultdict(int))
        }
        
        # Display game setup
        log_and_print("\nGame Starting! Meet our players:")
        for agent in agents:
            log_and_print(f"\n{agent.name}'s persona: {agent.persona}")
            traits = [trait for trait, value in agent.personality_traits.items() if value]
            log_and_print(f"Personality traits: {', '.join(traits)}")
        log_and_print("\n" + "="*80 + "\n")
        
        # Main game loop
        done = False
        current_score = 0
        prev_life_tokens = 3  # Starting life tokens
        
        while not done:
            for agent_id, agent in enumerate(agents):
                observation = observations['player_observations'][agent_id]
                
                if observation['current_player'] == agent_id:
                    game_stats['turns'] += 1
                    current_score = sum(observation['fireworks'].values())
                    game_stats['score_progression'].append(current_score)
                    
                    # Track life token changes
                    current_life_tokens = observation['life_tokens']
                    if current_life_tokens < prev_life_tokens:
                        game_stats['lives_lost'] += 1
                    prev_life_tokens = current_life_tokens
                    
                    # Log turn header with detailed state
                    log_and_print(f"\nTurn {game_stats['turns']}: {agent.name}'s turn")
                    log_and_print(f"Current Score: {current_score}")
                    log_and_print(format_game_state(observation, agents))
                    
                    # Get agent's move
                    action = agent.act(observation)
                    
                    # Update statistics based on action
                    action_type = action['action_type']
                    game_stats['player_actions'][agent.name][action_type] += 1
                    
                    if action_type == 'PLAY':
                        game_stats['cards_played'] += 1
                    elif action_type == 'DISCARD':
                        game_stats['cards_discarded'] += 1
                    elif action_type.startswith('REVEAL'):
                        game_stats['hints_given'] += 1
                       
                    # Execute move and get new state
                    observations, reward, done, _ = environment.step(action)
                    
                    # Format and log action results
                    log_and_print("\nAction Summary:")
                    if action_type == 'PLAY':
                        new_score = sum(observations['player_observations'][0]['fireworks'].values())
                        success = new_score > current_score
                        result = "successfully" if success else "unsuccessfully"
                        log_and_print(f"Played card {action['card_index']} {result}")
                        log_and_print(f"New score: {new_score}")
                    elif action_type == 'DISCARD':
                        log_and_print(f"Discarded card {action['card_index']}")
                        log_and_print(f"Information tokens: {observation['information_tokens']}")
                    else:
                        target = agent.get_player_name(observation['current_player'], action['target_offset'])
                        info_type = 'color' if action_type == 'REVEAL_COLOR' else 'rank'
                        info_value = action['color'] if info_type == 'color' else action['rank'] + 1
                        log_and_print(f"Gave hint to {target} about {info_type} {info_value}")
                    
                    log_and_print("-" * 40)
                    
                    if done:
                        break
        
        # Game complete - log final statistics
        final_score = max(0, current_score)
        
        log_and_print("\nGame Complete! Final Statistics:")
        log_and_print(f"Final Score: {final_score}")
        log_and_print(f"Total Turns: {game_stats['turns']}")
        log_and_print(f"Lives Lost: {game_stats['lives_lost']}")
        log_and_print(f"Hints Given: {game_stats['hints_given']}")
        log_and_print(f"Cards Played: {game_stats['cards_played']}")
        log_and_print(f"Cards Discarded: {game_stats['cards_discarded']}")
        
        log_and_print("\nPlayer Action Summary:")
        for player, actions in game_stats['player_actions'].items():
            log_and_print(f"\n{player}:")
            for action_type, count in actions.items():
                log_and_print(f"- {action_type}: {count}")
        
        log_and_print(f"\nScore Progression: {game_stats['score_progression']}")
        log_and_print(f"\nGame log saved to: {log_filename}")
        
        return final_score

if __name__ == "__main__":
    score = run_game()