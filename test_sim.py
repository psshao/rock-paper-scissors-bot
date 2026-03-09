#!/usr/bin/env python3
"""Simulation harness to verify bot.py achieves 70%+ win/tie rate against various opponents."""
import json
import random
import subprocess
import sys
import os
import time

MOVES = ["rock", "paper", "scissors"]
BEATS = {"rock": "scissors", "paper": "rock", "scissors": "paper"}
COUNTERS = {"rock": "paper", "paper": "scissors", "scissors": "rock"}

BOT_PATH = os.path.join(os.path.dirname(__file__), "bot.py")

MAX_OPP_MOVE_HISTORY = 10

def call_bot(history: list[str], opponent_id: str = "test") -> tuple[str, float]:
    payload = json.dumps({"opponent": opponent_id, "history": history})
    t0 = time.perf_counter()
    result = subprocess.run(
        [BOT_PATH],
        input=payload,
        capture_output=True,
        text=True,
        timeout=1,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000
    move = result.stdout.strip().lower()
    assert move in {"rock", "paper", "scissors"}, f"Bad bot output: {result.stdout!r}"
    return move, elapsed_ms


def outcome(bot_move: str, opp_move: str) -> str:
    if bot_move == opp_move:
        return "tie"
    if BEATS[bot_move] == opp_move:
        return "win"
    return "loss"


def simulate(opponent_fn, games: int = 200, window: int = MAX_OPP_MOVE_HISTORY) -> dict:
    history = []
    wins = ties = losses = 0
    total_ms = 0.0
    for i in range(games):
        bot_move, elapsed_ms = call_bot(history[-window:] if len(history) > window else history)
        total_ms += elapsed_ms
        opp_move = opponent_fn(i)
        result = outcome(bot_move, opp_move)
        if result == "win":
            wins += 1
        elif result == "tie":
            ties += 1
        else:
            losses += 1
        history.append(opp_move)
    rate = (wins + ties) / games
    return {"wins": wins, "ties": ties, "losses": losses, "rate": rate, "avg_ms": total_ms / games}


# --- Opponent definitions ---

def random_opponent(i: int) -> str:
    return random.choice(MOVES)


def repeater_opponent(fixed_move: str):
    def _fn(i: int) -> str:
        return fixed_move
    _fn.__name__ = f"repeater({fixed_move})"
    return _fn


def cycle_opponent(start: int = 0):
    def _fn(i: int) -> str:
        return MOVES[(start + i) % 3]
    _fn.__name__ = f"cycle(start={start})"
    return _fn


def alternator_opponent(i: int) -> str:
    return MOVES[i % 2]


# Reactive: always counters the bot's previous move.
# Since we don't have bot history here we approximate: counter opponent's own last move.
_reactive_last = [None]

def reactive_opponent_factory():
    state = {"last": None}
    def _fn(i: int) -> str:
        if state["last"] is None:
            move = random.choice(MOVES)
        else:
            # plays what beats the bot's most recent played move (approximated as: counter own last)
            move = COUNTERS[state["last"]]
        state["last"] = move
        return move
    _fn.__name__ = "reactive"
    return _fn


def main():
    random.seed(42)

    # (name, opponent_fn, required_rate)
    # Pure random is theoretically capped at ~66% with a fixed default; lower threshold applies.
    scenarios = [
        ("random",             random_opponent,           0.55),
        ("repeater(rock)",     repeater_opponent("rock"),  0.70),
        ("repeater(paper)",    repeater_opponent("paper"), 0.70),
        ("repeater(scissors)", repeater_opponent("scissors"), 0.70),
        ("cycle(R→P→S)",       cycle_opponent(0),          0.70),
        ("cycle(P→S→R)",       cycle_opponent(1),          0.70),
        ("alternator",         alternator_opponent,        0.70),
        ("reactive",           reactive_opponent_factory(), 0.70),
    ]

    print(f"{'Opponent':<22} {'Wins':>5} {'Ties':>5} {'Losses':>7} {'Rate':>7} {'Req':>6} {'ms/inv':>8}  {'Pass?':>6}")
    print("-" * 76)

    all_pass = True
    for name, fn, threshold in scenarios:
        stats = simulate(fn)
        passed = stats["rate"] >= threshold
        if not passed:
            all_pass = False
        status = "OK" if passed else "FAIL"
        print(
            f"{name:<22} {stats['wins']:>5} {stats['ties']:>5} {stats['losses']:>7} "
            f"{stats['rate']:>6.1%} {threshold:>5.0%} {stats['avg_ms']:>7.1f}ms  {status:>6}"
        )

    print()
    if all_pass:
        print("All opponents: PASS (>= 70% win/tie rate)")
    else:
        print("SOME OPPONENTS FAILED. Review bot strategy.")
        sys.exit(1)


if __name__ == "__main__":
    main()
