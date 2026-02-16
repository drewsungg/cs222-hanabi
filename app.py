"""Flask application — serves the Hanabi game UI and streams turns via SSE."""

import json

from flask import Flask, render_template, request, jsonify, Response

from hanabi_engine import HanabiGame, COLORS, COLOR_NAMES
from agents import HanabiLLMAgent
from persona_loader import load_players
from llm_utils import create_client
from config import LLM_PROVIDER, OPENAI_API_KEY, ANTHROPIC_API_KEY

app = Flask(__name__)

# In-memory state (single-user localhost)
_state: dict = {}


def _init_client():
    """Create LLM client from .env config on first use."""
    if "client" in _state:
        return
    provider = LLM_PROVIDER
    api_key = OPENAI_API_KEY if provider == "openai" else ANTHROPIC_API_KEY
    if not api_key:
        raise RuntimeError(
            f"No API key found for provider '{provider}'. "
            "Set OPENAI_API_KEY or ANTHROPIC_API_KEY in your .env file."
        )
    _state["client"] = create_client(provider, api_key)
    _state["provider"] = provider


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/game/start", methods=["POST"])
def game_start():
    """Create a new game + agents, return player info."""
    try:
        _init_client()
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 400

    data = request.get_json(force=True) if request.is_json else {}
    num_players = data.get("num_players", 5)
    num_players = max(2, min(5, int(num_players)))

    game = HanabiGame(num_players=num_players)
    players = load_players(num_players)
    agents = []
    for p in players:
        agent = HanabiLLMAgent(
            name=p["name"],
            persona=p["persona"],
            client=_state["client"],
            provider=_state["provider"],
        )
        agents.append(agent)

    names = [a.name for a in agents]
    for a in agents:
        a.all_player_names = names

    _state["game"] = game
    _state["agents"] = agents

    return jsonify({
        "players": [
            {"name": a.name, "persona": a.persona}
            for a in agents
        ],
        "num_players": num_players,
        "hand_size": game.hand_size,
    })


@app.route("/api/game/stream")
def game_stream():
    """SSE endpoint — streams one event per turn."""
    if "game" not in _state:
        return jsonify({"error": "No game in progress"}), 400

    game: HanabiGame = _state["game"]
    agents: list[HanabiLLMAgent] = _state["agents"]

    def generate_events():
        # Send initial state
        yield _sse(event="init", data=_snapshot(game, agents, turn=0))

        turn = 0
        while not game.done:
            pid = game.current_player
            obs = game.get_observation(pid)
            agent = agents[pid]

            action, meta = agent.act(obs)
            result = game.step(action)
            turn += 1

            event_data = {
                "turn": turn,
                "player": agent.name,
                "player_index": pid,
                "action": _describe_action(action, obs, agents),
                "action_raw": action,
                "reasoning": meta.get("llm_response") or meta.get("reasoning", ""),
                "method": meta.get("method", ""),
                "confidence": meta.get("confidence"),
                "success": result["info"].get("success"),
                "card": result["info"].get("card"),
                **_snapshot(game, agents, turn),
            }
            yield _sse(event="turn", data=event_data)

        # Game over
        yield _sse(event="gameover", data={
            "score": game.score(),
            "fireworks": game.fireworks,
            "life_tokens": game.life_tokens,
            "turns": turn,
        })

    return Response(
        generate_events(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _snapshot(game: HanabiGame, agents, turn: int) -> dict:
    """Current board state visible to a spectator (all cards face-up)."""
    hands = []
    for pid in range(game.num_players):
        cards = []
        for i, card in enumerate(game.hands[pid]):
            ck = game.card_knowledge[pid][i]
            cards.append({
                "color": card.color if card else None,
                "rank": card.rank if card else None,
                "known_color": ck["color"],
                "known_rank": ck["rank"],
            })
        hands.append({"name": agents[pid].name, "cards": cards})

    return {
        "hands": hands,
        "fireworks": dict(game.fireworks),
        "life_tokens": game.life_tokens,
        "info_tokens": game.info_tokens,
        "deck_size": len(game.deck),
        "discard_pile": [c.to_dict() for c in game.discard_pile],
        "score": game.score(),
        "turn": turn,
        "current_player": game.current_player,
    }


def _describe_action(action: dict, obs: dict, agents: list) -> str:
    """Human-readable action description."""
    at = action["action_type"]
    cur = obs["current_player"]
    n = len(agents)
    if at == "PLAY":
        return f"Play card {action['card_index']}"
    if at == "DISCARD":
        return f"Discard card {action['card_index']}"
    if at == "REVEAL_COLOR":
        target = agents[(cur + action["target_offset"]) % n].name
        return f"Tell {target} about color {action['color']}"
    if at == "REVEAL_RANK":
        target = agents[(cur + action["target_offset"]) % n].name
        return f"Tell {target} about rank {action['rank'] + 1}"
    return str(action)


def _sse(event: str, data) -> str:
    payload = json.dumps(data, default=str)
    return f"event: {event}\ndata: {payload}\n\n"


if __name__ == "__main__":
    app.run(debug=True, port=5001, threaded=True)
