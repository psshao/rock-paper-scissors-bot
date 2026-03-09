#!/usr/bin/env python3
"""
Generate plain-text input files for testing bot.py against different opponent strategies.

Each file contains one move per line (the opponent's history, oldest first, max 10 moves).
Files are numbered sequentially and contain no strategy information in their name.
Pipe any file to bot.py:

    ./bot --input inputs/001.txt

Usage:
    python3 gen_inputs.py [--strategy STRATEGY] [--length N] [--out DIR] [--seed S]
    python3 gen_inputs.py --all
    python3 gen_inputs.py --list

History length is capped at 10 (problem-statement constraint).
"""

import argparse
import inspect
import random
import subprocess
import sys
from pathlib import Path

MOVES = ["rock", "paper", "scissors"]
COUNTERS = {"rock": "paper", "paper": "scissors", "scissors": "rock"}
MAX_HISTORY = 10

# ---------------------------------------------------------------------------
# Strategy generators — each returns a list[str] of opponent moves
# ---------------------------------------------------------------------------

def gen_random(length: int, seed: int | None = None) -> list[str]:
    """Uniformly random — no exploitable pattern."""
    rng = random.Random(seed)
    return [rng.choice(MOVES) for _ in range(length)]


def gen_repeater(length: int, move: str = "rock") -> list[str]:
    """Always plays the same move."""
    return [move] * length


def gen_cycler(length: int, start: int = 0) -> list[str]:
    """Cycles through rock → paper → scissors in order."""
    return [MOVES[(start + i) % 3] for i in range(length)]


def gen_alternator(length: int, move_a: str = "rock", move_b: str = "paper") -> list[str]:
    """Alternates between two fixed moves."""
    return [[move_a, move_b][i % 2] for i in range(length)]


def gen_reactive(length: int, bot_seed_move: str = "paper") -> list[str]:
    """
    Simulates an opponent that counters the bot's previous move.
    Runs bot.py after each round to track what it would actually play.
    """
    history: list[str] = []
    bot_prev = bot_seed_move

    import tempfile, os
    bot_path = str(Path(__file__).parent / "bot")
    for _ in range(length):
        opp_move = COUNTERS[bot_prev]
        history.append(opp_move)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as tmp:
            tmp.write("\n".join(history))
            tmp_path = tmp.name
        try:
            result = subprocess.run(
                [bot_path, "--input", tmp_path],
                capture_output=True, text=True, timeout=1,
            )
            bot_prev = result.stdout.strip().lower()
            if bot_prev not in MOVES:
                bot_prev = "paper"
        except Exception:
            bot_prev = "paper"
        finally:
            os.unlink(tmp_path)

    return history


def gen_markov(length: int, seed: int | None = None) -> list[str]:
    """
    Biased 1st-order Markov chain — harder for cycle detection, targets N-gram models.

    Transitions: rock→paper 70%, paper→scissors 70%, scissors→rock 70% (with noise).
    """
    transitions = {
        "rock":     [("rock", 0.10), ("paper", 0.70), ("scissors", 0.20)],
        "paper":    [("rock", 0.20), ("paper", 0.10), ("scissors", 0.70)],
        "scissors": [("rock", 0.70), ("paper", 0.20), ("scissors", 0.10)],
    }
    rng = random.Random(seed)
    current = rng.choice(MOVES)
    history = []
    for _ in range(length):
        history.append(current)
        choices, weights = zip(*transitions[current])
        current = rng.choices(choices, weights=weights, k=1)[0]
    return history


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

STRATEGIES: dict[str, tuple] = {
    # name: (fn, kwargs, description)
    "random":               (gen_random,     {},                                         "Uniformly random — no pattern"),
    "repeater-rock":        (gen_repeater,   {"move": "rock"},                           "Always plays rock"),
    "repeater-paper":       (gen_repeater,   {"move": "paper"},                          "Always plays paper"),
    "repeater-scissors":    (gen_repeater,   {"move": "scissors"},                       "Always plays scissors"),
    "cycle-rps":            (gen_cycler,     {"start": 0},                               "Cycles rock→paper→scissors"),
    "cycle-psr":            (gen_cycler,     {"start": 1},                               "Cycles paper→scissors→rock"),
    "cycle-srp":            (gen_cycler,     {"start": 2},                               "Cycles scissors→rock→paper"),
    "alternator-rp":        (gen_alternator, {"move_a": "rock",  "move_b": "paper"},     "Alternates rock/paper"),
    "alternator-rs":        (gen_alternator, {"move_a": "rock",  "move_b": "scissors"},  "Alternates rock/scissors"),
    "alternator-ps":        (gen_alternator, {"move_a": "paper", "move_b": "scissors"},  "Alternates paper/scissors"),
    "reactive":             (gen_reactive,   {},                                         "Counters the bot's previous move (simulated)"),
    "markov":               (gen_markov,     {},                                         "Biased 1st-order Markov chain"),
}


# ---------------------------------------------------------------------------
# File writing
# ---------------------------------------------------------------------------

def write_input(out_dir: Path, strategy: str, history: list[str]) -> Path:
    path = out_dir / f"{strategy}.txt"
    path.write_text("\n".join(history) + ("\n" if history else ""))
    return path


def generate(strategy: str, length: int, out_dir: Path, seed: int | None = None) -> Path:
    length = min(length, MAX_HISTORY)
    fn, kwargs, _ = STRATEGIES[strategy]

    call_kwargs = dict(kwargs)
    if "seed" in inspect.signature(fn).parameters:
        call_kwargs["seed"] = seed

    history = fn(length=length, **call_kwargs)
    return write_input(out_dir, strategy, history)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def cmd_list(_args):
    print(f"{'Strategy':<22}  Description")
    print("-" * 60)
    for name, (_, _, desc) in STRATEGIES.items():
        print(f"  {name:<22} {desc}")


def cmd_generate(args):
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    targets = list(STRATEGIES) if args.all else [args.strategy]

    generated = []
    for strategy in targets:
        if strategy not in STRATEGIES:
            print(f"Unknown strategy '{strategy}'. Use --list to see options.", file=sys.stderr)
            sys.exit(1)
        path = generate(strategy, args.length, out_dir, seed=args.seed)
        generated.append(path)
        print(f"  wrote {path}")

    print()
    print("Test with:")
    for path in generated:
        print(f"  ./bot --input {path}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--list", action="store_true", help="List available strategies and exit")
    parser.add_argument("--all", action="store_true", help="Generate files for all strategies")
    parser.add_argument("--strategy", "-s", metavar="NAME", help="Strategy name (see --list)")
    parser.add_argument("--length", "-n", type=int, default=MAX_HISTORY,
                        metavar="N", help=f"History length (default/max: {MAX_HISTORY})")
    parser.add_argument("--out", "-o", default="inputs",
                        metavar="DIR", help="Output directory (default: inputs/)")
    parser.add_argument("--seed", type=int, default=None,
                        metavar="S", help="Random seed for reproducible output")

    args = parser.parse_args()

    if args.list:
        cmd_list(args)
        return

    if not args.all and not args.strategy:
        parser.error("Provide --strategy NAME or --all")

    cmd_generate(args)


if __name__ == "__main__":
    main()
