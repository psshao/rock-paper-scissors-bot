#!/usr/bin/env python3
import argparse
import json
import sys
from collections import Counter

COUNTERS = {"rock": "paper", "paper": "scissors", "scissors": "rock"}
VALID = {"rock", "paper", "scissors"}

def parse_input(input_file: str | None) -> list[str]:
    """Read and sanitize opponent history from a file or stdin.

    The caller may supply a plain-text file containing one move per line.  In
    that case the function reads each line, normalizes to lowercase, and
    retains only valid moves (`rock`, `paper`, `scissors`).

    If `input_file` is ``None`` the function instead expects JSON on stdin of
    the form ``{"opponent": "<id>", "history": [...]}``.  The ``history``
    array is filtered in the same way.

    The returned list contains only moves from the opponent; any extraneous
    data or malformed entries are silently dropped.  An empty list is
    returned on error or if no valid moves are found.

    Args:
        input_file: Path to a plain-text history file, or ``None`` to read
            structured data from stdin.

    Returns:
        A cleaned list of past opponent moves ready for analysis.
    """
    if input_file:
        with open(input_file) as f:
            return [m for line in f if (m := line.strip().lower()) in VALID]
    # JSON from stdin: {"opponent": "...", "history": [...]}
    data = json.loads(sys.stdin.read())
    return [m for m in data.get("history", []) if m in VALID]

def detect_cycle(h: list[str]) -> str | None:
    """Detects a simple repeating cycle in the opponent's history.

    The function checks for periodic behaviour with periods 1, 2, and 3.  A
    period *p* is considered valid only when the entire history matches the
    last *p* entries repeated.  This strict check avoids false positives on
    short sequences.  Once a cycle is confirmed, the next move of the cycle is
    predicted and the corresponding counter move is returned.

    Args:
        h: Sequence of past opponent moves.  Older moves come first.

    Returns:
        The move that beats the anticipated next entry in the detected cycle,
        or ``None`` if no consistent cycle was found.  Periods beyond 3 are not
        evaluated to keep computation trivial and because longer cycles require
        more history than the game limit allows.
    """
    for p in [1, 2, 3]:
        if len(h) < 2 * p:
            continue
        pattern = h[-p:]
        if all(h[i] == pattern[i % p] for i in range(len(h))):
            return COUNTERS[pattern[len(h) % p]]
    return None

def ngram_predict(h: list[str], n: int) -> str | None:
    """Predict the opponent's next move using an n‑gram model.

    This function builds a simple frequency table of what move follows each
    length- *n* sequence in the opponent's `history` list.  It then looks up the
    most recent *n*-gram and, if the observed successor is confident enough,
    returns the **counter** to that move.

    The model is only considered reliable when:
    1. We have at least *n+1* history entries (otherwise there is nothing to
       learn).
    2. The most common next-move for the current n-gram has appeared at least
       twice in the data.
    3. There is a unique leader; ties between the top two candidates yield
       `None` since we cannot decide.

    Args:
        h: List of past opponent moves (each "rock", "paper" or
            "scissors").  Earlier entries are older.
        n: Size of the n-gram to use (1 for unigram, 2 for bigram, etc.).

    Returns:
        The move that beats the predicted successor (via :data:`COUNTERS`), or
        `None` if the model isn't confident enough to make a prediction.
    """
    if len(h) < n + 1:
        return None
    counts: dict = {}
    for i in range(len(h) - n):
        key = tuple(h[i:i+n])
        counts.setdefault(key, Counter())[h[i+n]] += 1
    key = tuple(h[-n:])
    if key not in counts:
        return None
    top = counts[key].most_common(2)
    if top[0][1] < 2:
        return None
    if len(top) > 1 and top[0][1] == top[1][1]:
        return None
    return COUNTERS[top[0][0]]

def frequency_bias(h: list[str]) -> str | None:
    """Exploit a strong move frequency bias in the opponent's history.

    If one of the three moves has been played significantly more often than
    the others (at least 40% of the time and with at least five total moves),
    we assume the opponent favors that move and counter it directly.  This is a
    low-confidence heuristic used only when more sophisticated strategies have
    failed to produce a prediction.

    Args:
        h: List of past opponent moves.

    Returns:
        The move that beats the opponent's most frequent move, or ``None`` if
        the history is too short or no move exceeds the threshold.
    """
    if len(h) < 5:
        return None
    freq = Counter(h)
    move, count = freq.most_common(1)[0]
    if count / len(h) < 0.40:
        return None
    return COUNTERS[move]

def choose_move(h: list[str]) -> str:
    return (detect_cycle(h)
            or ngram_predict(h, 2)
            or ngram_predict(h, 1)
            or frequency_bias(h)
            or "paper")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rock Paper Scissors bot")
    parser.add_argument("--input", metavar="FILE",
                        help="Plain-text history file (one move per line). "
                             "If omitted, reads JSON from stdin.")
    args = parser.parse_args()

    try:
        history = parse_input(args.input)
    except Exception:
        history = []
    print(choose_move(history))
