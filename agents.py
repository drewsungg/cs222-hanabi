"""LLM-powered Hanabi agent.

Ported from hanabi_rulebased_reasoning.py with cleanups:
- No C++ dependency
- Supports OpenAI + Anthropic via llm_utils
- act() returns (action_dict, metadata_dict)
"""

import re
import random
from collections import defaultdict

from hanabi_engine import COLORS, RANK_COUNTS
from llm_utils import generate


class HanabiLLMAgent:
    """An LLM-enhanced Hanabi player with persona-driven behaviour."""

    def __init__(self, name: str, persona: str, client, provider: str):
        self.name = name
        self.persona = persona
        self.client = client
        self.provider = provider
        self.all_player_names: list[str] = []

        self.personality_traits = self._extract_personality_traits()
        self.play_history: list[dict] = []
        self.success_rate: dict[str, dict] = {}

    # ------------------------------------------------------------------
    # Personality
    # ------------------------------------------------------------------
    def _extract_personality_traits(self) -> dict:
        p = str(self.persona).lower()
        return {
            "analytical": any(t in p for t in ["analytical", "logical", "thoughtful"]),
            "social": any(t in p for t in ["helping others", "empathy", "friendly"]),
            "risk_taking": any(t in p for t in ["confident", "ambitious", "outgoing"]),
            "perfectionist": any(t in p for t in ["detail-oriented", "perfectionist", "meticulous"]),
            "intuitive": any(t in p for t in ["intuitive", "gut feeling", "instinct"]),
            "methodical": any(t in p for t in ["systematic", "organized", "structured"]),
        }

    def _get_personality_biases(self) -> dict:
        biases = {"PLAY": 0.0, "DISCARD": 0.0, "HINT": 0.0}
        if self.personality_traits["risk_taking"]:
            biases["PLAY"] += 0.4
        if self.personality_traits["analytical"]:
            biases["DISCARD"] += 0.05
        if self.personality_traits["social"]:
            biases["HINT"] += 0.08
        return biases

    # ------------------------------------------------------------------
    # Card helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _is_card_playable(card: dict, fireworks: dict) -> bool:
        if card.get("color") is None or card.get("rank") is None:
            return False
        return fireworks[card["color"]] == card["rank"]

    @staticmethod
    def _is_last_copy(observation: dict, card: dict) -> bool:
        if not card.get("color") or card.get("rank") is None:
            return False
        discarded = sum(
            1
            for d in observation["discard_pile"]
            if d["color"] == card["color"] and d["rank"] == card["rank"]
        )
        max_copies = RANK_COUNTS.get(card["rank"], 2)
        return discarded >= max_copies - 1

    # ------------------------------------------------------------------
    # Game-state evaluation helpers
    # ------------------------------------------------------------------
    def _analyze_discard_safety(self, observation: dict, card_index: int) -> float:
        card_knowledge = observation["card_knowledge"][0][card_index]
        fireworks = observation["fireworks"]

        if card_knowledge["color"] is not None or card_knowledge["rank"] is not None:
            return 0.0  # hinted — don't discard

        return 0.7  # unknown card is moderately safe

    def _predict_play_success(self, card_knowledge: dict, observation: dict) -> float:
        key = f"{card_knowledge.get('color', 'unk')}_{card_knowledge.get('rank', 'unk')}"
        if key in self.success_rate and self.success_rate[key]["total"] > 0:
            return self.success_rate[key]["success"] / self.success_rate[key]["total"]
        return 0.5

    def _calculate_risk_tolerance(self, observation: dict) -> float:
        life = observation["life_tokens"]
        info = observation["information_tokens"]
        fireworks = observation["fireworks"]
        progress = sum(fireworks.values()) / 25.0
        risk = 0.9

        if life == 1:
            risk *= 0.4
        elif life == 2:
            risk *= 0.8

        if progress < 0.4:
            risk *= 2.0
        elif progress > 0.8:
            risk *= 0.8

        if info <= 2:
            risk *= 0.8
        elif info >= 7:
            risk *= 1.1

        if self.personality_traits["risk_taking"]:
            risk *= 1.2
        if self.personality_traits["perfectionist"]:
            risk *= 0.8

        if life == 1 or info == 0 or sum(fireworks.values()) > 20:
            risk = min(risk, 0.5)

        return max(0.1, min(5.0, risk))

    def _evaluate_hint_value(self, move: dict, observation: dict) -> float:
        if not move["action_type"].startswith("REVEAL"):
            return 0.0
        offset = move["target_offset"]
        hand = observation["observed_hands"][offset]
        fireworks = observation["fireworks"]
        knowledge = observation["card_knowledge"][offset]
        value = 0.0
        new_info = False

        for idx, card in enumerate(hand):
            if card is None:
                continue
            if move["action_type"] == "REVEAL_COLOR" and card["color"] == move["color"]:
                if knowledge[idx]["color"] is None:
                    new_info = True
                if fireworks[card["color"]] == card["rank"]:
                    value += 2.0
                elif card["rank"] > 0 and fireworks[card["color"]] == card["rank"] - 1:
                    value += 1.0
            elif move["action_type"] == "REVEAL_RANK" and card["rank"] == move["rank"]:
                if knowledge[idx]["rank"] is None:
                    new_info = True
                if fireworks[card["color"]] == card["rank"]:
                    value += 2.0
                elif card["rank"] > 0 and fireworks[card["color"]] == card["rank"] - 1:
                    value += 1.0

        if new_info:
            value *= 1.5
        return value

    def _evaluate_strict_rules(self, observation: dict, legal_moves: list[dict]):
        """Return (move, reason) if a strict rule applies, else (None, None)."""
        fireworks = observation["fireworks"]
        known = observation["card_knowledge"][0]

        # Play any card that is confirmed playable
        for m in legal_moves:
            if m["action_type"] == "PLAY":
                c = known[m["card_index"]]
                if c["color"] is not None and c["rank"] is not None:
                    if fireworks[c["color"]] == c["rank"]:
                        return m, "Confirmed playable card"
        return None, None

    # ------------------------------------------------------------------
    # Formatting for the LLM prompt
    # ------------------------------------------------------------------
    def _get_player_name(self, current: int, offset: int) -> str:
        if not self.all_player_names:
            return f"Player {offset}"
        return self.all_player_names[(current + offset) % len(self.all_player_names)]

    def _format_known_cards(self, observation: dict) -> str:
        lines = []
        for i, ck in enumerate(observation["card_knowledge"][0]):
            parts = []
            if ck["color"] is not None:
                parts.append(f"color: {ck['color']}")
            if ck["rank"] is not None:
                parts.append(f"rank: {ck['rank'] + 1}")
            lines.append(f"Card {i}: {', '.join(parts) if parts else 'No information'}")
        return "\n".join(lines)

    def _format_legal_moves(self, legal_moves: list[dict], observation: dict) -> str:
        lines = []
        cur = observation["current_player"]
        for i, m in enumerate(legal_moves):
            if m["action_type"] == "PLAY":
                lines.append(f"{i}: Play card {m['card_index']}")
            elif m["action_type"] == "DISCARD":
                lines.append(f"{i}: Discard card {m['card_index']}")
            elif m["action_type"].startswith("REVEAL"):
                name = self._get_player_name(cur, m["target_offset"])
                kind = "color" if m["action_type"] == "REVEAL_COLOR" else "rank"
                val = m["color"] if kind == "color" else m["rank"] + 1
                lines.append(f"{i}: Tell {name} about {kind} {val}")
        return "\n".join(lines)

    def _format_move(self, move: dict) -> str:
        if move["action_type"] == "PLAY":
            return f"Play card {move['card_index']}"
        if move["action_type"] == "DISCARD":
            return f"Discard card {move['card_index']}"
        if move["action_type"].startswith("REVEAL"):
            kind = "color" if move["action_type"] == "REVEAL_COLOR" else "rank"
            val = move.get("color") if kind == "color" else move.get("rank", -1) + 1
            return f"Hint about {kind} {val}"
        return str(move)

    def _analyze_game_state(self, observation: dict) -> str:
        life = observation["life_tokens"]
        info = observation["information_tokens"]
        fw = observation["fireworks"]
        parts = []
        if life == 1:
            parts.append("Critical — one life remaining, play safely.")
        elif life == 2:
            parts.append("Caution — two lives left.")
        if info == 0:
            parts.append("Must discard to regain info tokens.")
        elif info == 8:
            parts.append("Full info tokens — use them for coordination.")
        score = sum(fw.values())
        if score == 0:
            parts.append("Need to start building stacks.")
        elif score > 20:
            parts.append("Good progress — focus on finishing stacks.")
        return " ".join(parts) if parts else "Standard game state."

    # ------------------------------------------------------------------
    # Main decision method
    # ------------------------------------------------------------------
    def act(self, observation: dict) -> tuple[dict, dict]:
        """Choose a move. Returns (action_dict, metadata_dict)."""
        legal_moves = observation["legal_moves"]
        fireworks = observation["fireworks"]
        info_tokens = observation["information_tokens"]
        life_tokens = observation["life_tokens"]

        metadata: dict = {"player": self.name, "persona": self.persona}

        # 1. Strict rules
        strict, reason = self._evaluate_strict_rules(observation, legal_moves)
        if strict:
            metadata["reasoning"] = reason
            metadata["method"] = "strict_rule"
            return strict, metadata

        # 2. Score every legal move
        move_scores: list[tuple[dict, float, list[str]]] = []

        for m in legal_moves:
            if m["action_type"] == "PLAY":
                ck = observation["card_knowledge"][0][m["card_index"]]
                pred = self._predict_play_success(ck, observation)
                risk = self._calculate_risk_tolerance(observation)
                score = pred * risk
                move_scores.append((m, score, [f"play success={pred:.2f}, risk={risk:.2f}"]))
            elif m["action_type"] == "DISCARD":
                safety = self._analyze_discard_safety(observation, m["card_index"])
                if info_tokens == 0:
                    safety *= 2.0
                elif info_tokens >= 7:
                    safety *= 0.5
                move_scores.append((m, safety, [f"discard safety={safety:.2f}"]))
            elif m["action_type"].startswith("REVEAL"):
                hv = self._evaluate_hint_value(m, observation)
                move_scores.append((m, hv, [f"hint value={hv:.2f}"]))

        # 3. Build context for LLM
        game_analysis = self._analyze_game_state(observation)
        context = (
            f"Game Analysis: {game_analysis}\n"
            f"Fireworks: {fireworks}, Info tokens: {info_tokens}, Life tokens: {life_tokens}\n\n"
            f"My cards:\n{self._format_known_cards(observation)}\n\n"
            f"Available Moves:\n{self._format_legal_moves(legal_moves, observation)}\n\n"
            f"Move scores (pre-computed): {[(self._format_move(m), f'{s:.2f}') for m, s, _ in move_scores]}\n\n"
            "Based on the information and your persona, choose the most optimal move.\n"
            "Your response must follow this strict format:\n"
            "**MOVE**: <index> (short description)\n"
            "**CONFIDENCE**: <value between 0 and 1>\n"
        )
        prompt = (
            f"As {self.name}, a Hanabi player who {self.persona}, analyze the turn.\n\n"
            f"{context}"
        )

        # 4. Call LLM
        try:
            messages = [
                {"role": "system", "content": "You are a strategic Hanabi player focused on teamwork."},
                {"role": "user", "content": prompt},
            ]
            content = generate(
                self.client,
                self.provider,
                messages,
                temperature=0.4,
                max_tokens=800,
            )
            metadata["llm_response"] = content

            # Parse response
            move_match = re.search(r"\*\*MOVE\*\*:\s*(\d+)", content)
            conf_match = re.search(r"\*\*CONFIDENCE\*\*:\s*([\d.]+)", content)
            move_idx = int(move_match.group(1)) if move_match else None
            confidence = float(conf_match.group(1)) if conf_match else None

            if move_idx is not None and 0 <= move_idx < len(legal_moves):
                chosen = legal_moves[move_idx]
                # Validate hint moves
                if chosen["action_type"].startswith("REVEAL"):
                    offset = chosen["target_offset"]
                    hand = observation["observed_hands"][offset]
                    if chosen["action_type"] == "REVEAL_COLOR":
                        ok = any(c and c["color"] == chosen["color"] for c in hand)
                    else:
                        ok = any(c and c["rank"] == chosen["rank"] for c in hand)
                    if not ok:
                        raise ValueError("LLM chose invalid hint")
                metadata["reasoning"] = content
                metadata["confidence"] = confidence
                metadata["method"] = "llm"
                return chosen, metadata

            raise ValueError(f"Invalid move index: {move_idx}")

        except Exception as e:
            metadata["llm_error"] = str(e)

        # 5. Fallback — pick best scored move
        if move_scores:
            best_move, best_score, reasons = max(move_scores, key=lambda x: x[1])
            metadata["reasoning"] = f"Fallback: {reasons}"
            metadata["method"] = "fallback_scored"
            return best_move, metadata

        # Absolute fallback
        metadata["reasoning"] = "No scores — random"
        metadata["method"] = "fallback_random"
        return random.choice(legal_moves), metadata
