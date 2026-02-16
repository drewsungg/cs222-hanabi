"""Pure Python Hanabi game engine.

Replaces the C++ hanabi-learning-environment dependency with a clean
implementation that produces observation dicts in the same format the
LLM agents expect.
"""

import random
from copy import deepcopy

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
COLORS = ["R", "Y", "G", "W", "B"]
COLOR_NAMES = {"R": "Red", "Y": "Yellow", "G": "Green", "W": "White", "B": "Blue"}
RANKS = [0, 1, 2, 3, 4]  # internal 0-indexed (display as 1-5)
RANK_COUNTS = {0: 3, 1: 2, 2: 2, 3: 2, 4: 1}  # how many copies per rank
MAX_INFO_TOKENS = 8
MAX_LIFE_TOKENS = 3
HAND_SIZE = {2: 5, 3: 5, 4: 4, 5: 4}


# ---------------------------------------------------------------------------
# Card helpers
# ---------------------------------------------------------------------------
class HanabiCard:
    __slots__ = ("color", "rank")

    def __init__(self, color: str, rank: int):
        self.color = color
        self.rank = rank

    def to_dict(self):
        return {"color": self.color, "rank": self.rank}

    def __repr__(self):
        return f"{self.color}{self.rank + 1}"


class HanabiDeck:
    def __init__(self):
        self.cards: list[HanabiCard] = []
        for color in COLORS:
            for rank in RANKS:
                for _ in range(RANK_COUNTS[rank]):
                    self.cards.append(HanabiCard(color, rank))
        random.shuffle(self.cards)

    def draw(self) -> HanabiCard | None:
        return self.cards.pop() if self.cards else None

    def __len__(self):
        return len(self.cards)


# ---------------------------------------------------------------------------
# Main game
# ---------------------------------------------------------------------------
class HanabiGame:
    """Full Hanabi game state machine."""

    def __init__(self, num_players: int = 5):
        assert 2 <= num_players <= 5
        self.num_players = num_players
        self.hand_size = HAND_SIZE[num_players]

        # State
        self.deck = HanabiDeck()
        self.hands: list[list[HanabiCard | None]] = [[] for _ in range(num_players)]
        self.card_knowledge: list[list[dict]] = [[] for _ in range(num_players)]
        self.fireworks = {c: 0 for c in COLORS}
        self.info_tokens = MAX_INFO_TOKENS
        self.life_tokens = MAX_LIFE_TOKENS
        self.discard_pile: list[HanabiCard] = []
        self.current_player = 0
        self.done = False
        self.final_round_player: int | None = None  # who triggered end
        self.turns_left: int | None = None

        # Deal
        for p in range(num_players):
            for _ in range(self.hand_size):
                card = self.deck.draw()
                self.hands[p].append(card)
                self.card_knowledge[p].append({"color": None, "rank": None})

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------
    def get_observation(self, player_id: int) -> dict:
        """Build the observation dict a specific player sees."""
        # observed_hands: offset-indexed (0 = self, hidden)
        observed_hands = []
        card_knowledge_obs = []
        for offset in range(self.num_players):
            actual = (player_id + offset) % self.num_players
            if offset == 0:
                # Player can't see own cards
                observed_hands.append([None] * len(self.hands[actual]))
            else:
                observed_hands.append(
                    [c.to_dict() if c else None for c in self.hands[actual]]
                )
            card_knowledge_obs.append(deepcopy(self.card_knowledge[actual]))

        legal_moves = self._legal_moves(player_id)

        return {
            "current_player": player_id,
            "current_player_offset": 0,
            "legal_moves": legal_moves,
            "fireworks": dict(self.fireworks),
            "life_tokens": self.life_tokens,
            "information_tokens": self.info_tokens,
            "observed_hands": observed_hands,
            "card_knowledge": card_knowledge_obs,
            "discard_pile": [c.to_dict() for c in self.discard_pile],
            "deck_size": len(self.deck),
            "num_players": self.num_players,
        }

    # ------------------------------------------------------------------
    # Legal moves
    # ------------------------------------------------------------------
    def _legal_moves(self, player_id: int) -> list[dict]:
        moves = []
        hand = self.hands[player_id]
        hand_len = len(hand)

        # PLAY and DISCARD for each card
        for i in range(hand_len):
            if hand[i] is not None:
                moves.append({"action_type": "PLAY", "card_index": i})
        if self.info_tokens < MAX_INFO_TOKENS:
            for i in range(hand_len):
                if hand[i] is not None:
                    moves.append({"action_type": "DISCARD", "card_index": i})

        # REVEAL hints (only if info tokens available)
        if self.info_tokens > 0:
            for offset in range(1, self.num_players):
                target = (player_id + offset) % self.num_players
                target_hand = self.hands[target]
                # Colors present in target hand
                colors_present = set()
                ranks_present = set()
                for card in target_hand:
                    if card is not None:
                        colors_present.add(card.color)
                        ranks_present.add(card.rank)
                for color in sorted(colors_present):
                    moves.append({
                        "action_type": "REVEAL_COLOR",
                        "target_offset": offset,
                        "color": color,
                    })
                for rank in sorted(ranks_present):
                    moves.append({
                        "action_type": "REVEAL_RANK",
                        "target_offset": offset,
                        "rank": rank,
                    })

        return moves

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------
    def step(self, action: dict) -> dict:
        """Apply action, return {reward, done, info}."""
        if self.done:
            return {"reward": 0, "done": True, "info": {}}

        player = self.current_player
        reward = 0
        info: dict = {"action": action, "player": player}
        action_type = action["action_type"]

        if action_type == "PLAY":
            idx = action["card_index"]
            card = self.hands[player][idx]
            info["card"] = card.to_dict() if card else None

            if card and self.fireworks[card.color] == card.rank:
                # Successful play
                self.fireworks[card.color] += 1
                reward = 1
                info["success"] = True
                # Bonus info token for completing a stack
                if self.fireworks[card.color] == 5 and self.info_tokens < MAX_INFO_TOKENS:
                    self.info_tokens += 1
            else:
                # Failed play
                if card:
                    self.discard_pile.append(card)
                self.life_tokens -= 1
                info["success"] = False
                if self.life_tokens <= 0:
                    self.done = True

            self._replace_card(player, idx)

        elif action_type == "DISCARD":
            idx = action["card_index"]
            card = self.hands[player][idx]
            info["card"] = card.to_dict() if card else None
            if card:
                self.discard_pile.append(card)
            self.info_tokens = min(self.info_tokens + 1, MAX_INFO_TOKENS)
            self._replace_card(player, idx)

        elif action_type == "REVEAL_COLOR":
            self.info_tokens -= 1
            offset = action["target_offset"]
            target = (player + offset) % self.num_players
            color = action["color"]
            info["target"] = target
            info["color"] = color
            affected = []
            for i, card in enumerate(self.hands[target]):
                if card and card.color == color:
                    self.card_knowledge[target][i]["color"] = color
                    affected.append(i)
            info["affected_cards"] = affected

        elif action_type == "REVEAL_RANK":
            self.info_tokens -= 1
            offset = action["target_offset"]
            target = (player + offset) % self.num_players
            rank = action["rank"]
            info["target"] = target
            info["rank"] = rank
            affected = []
            for i, card in enumerate(self.hands[target]):
                if card and card.rank == rank:
                    self.card_knowledge[target][i]["rank"] = rank
                    affected.append(i)
            info["affected_cards"] = affected

        # Check deck-empty endgame trigger
        if len(self.deck) == 0 and self.final_round_player is None:
            self.final_round_player = player
            self.turns_left = self.num_players

        # Advance turn
        self.current_player = (self.current_player + 1) % self.num_players

        # Check final round countdown
        if self.turns_left is not None:
            self.turns_left -= 1
            if self.turns_left <= 0:
                self.done = True

        score = sum(self.fireworks.values())
        return {"reward": reward, "done": self.done, "score": score, "info": info}

    def _replace_card(self, player: int, idx: int):
        card = self.deck.draw()
        self.hands[player][idx] = card
        self.card_knowledge[player][idx] = {"color": None, "rank": None}

    def score(self) -> int:
        return sum(self.fireworks.values())


# ---------------------------------------------------------------------------
# Simple random agent (for testing)
# ---------------------------------------------------------------------------
class SimpleAgent:
    """Plays a random legal move."""

    def __init__(self, name: str = "RandomBot"):
        self.name = name

    def act(self, observation: dict) -> tuple[dict, dict]:
        legal = observation["legal_moves"]
        move = random.choice(legal)
        return move, {"reasoning": "random"}


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
def self_test():
    """Run a complete game with random agents to verify engine correctness."""
    game = HanabiGame(num_players=5)
    agents = [SimpleAgent(f"Bot{i}") for i in range(5)]

    turn = 0
    while not game.done:
        pid = game.current_player
        obs = game.get_observation(pid)
        action, _ = agents[pid].act(obs)
        result = game.step(action)
        turn += 1

    final = game.score()
    print(f"Game finished in {turn} turns. Final score: {final}/25")
    print(f"Fireworks: {game.fireworks}")
    print(f"Lives remaining: {game.life_tokens}")
    print(f"Cards in discard: {len(game.discard_pile)}")

    # Basic sanity checks
    assert 0 <= final <= 25, f"Invalid score: {final}"
    assert game.life_tokens >= 0, "Negative lives"
    total_cards = sum(len([c for c in h if c]) for h in game.hands) + len(game.discard_pile) + len(game.deck)
    played = sum(game.fireworks.values())
    assert total_cards + played == 50, f"Card count mismatch: {total_cards + played}"
    print("All sanity checks passed!")
    return final


if __name__ == "__main__":
    # Run a few test games
    scores = []
    for i in range(5):
        print(f"\n--- Test Game {i+1} ---")
        scores.append(self_test())
    print(f"\nAverage score over 5 games: {sum(scores)/len(scores):.1f}")
