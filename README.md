# Rock Paper Scissors Bot

## Language & Dependencies

Python 3.11+. No external dependencies.

## Running the Bot
The bot can be provided an input file containing a newline delimited history of the opponent's previous choices, using the `--input` flag.

```bash
python3 bot.py --input test_inputs/random.txt
```

Or the bot reads a JSON payload from stdin and prints its move to stdout:

```bash
echo '{"opponent":"abc123","history":["rock","paper","scissors"]}' | python3 bot.py
```

**Example input file**
```text
rock
paper
scissors
rock
...
```

**Input schema**
```json
{"opponent": "<string>", "history": ["rock", "paper", "scissors", ...]}
```

**Output**: one of `rock`, `paper`, or `scissors` (single line to stdout).

## Unit Tests

```bash
python3 -m unittest discover -s unit_tests -v
```

or if you have `pytest` installed

```bash
pytest unit_tests/
```

## Quick Verification

```bash
# Empty history → paper (default)
echo '{"opponent":"test","history":[]}' | python3 bot.py

# Repeater (all rock) → paper
echo '{"opponent":"test","history":["rock","rock","rock","rock"]}' | python3 bot.py

# R→P→S cycler, next = rock → paper
echo '{"opponent":"test","history":["rock","paper","scissors","rock","paper","scissors"]}' | python3 bot.py

# Full simulation (200 games × 8 opponent types)
python3 test_sim.py
```

## Approach: Multi-Strategy Waterfall

Strategies are evaluated in priority order. The first one that fires with sufficient confidence wins.

### 1. Cycle Detection (highest priority)

Tests periods 1, 2, and 3. For period P:
- Requires `len(history) >= 2*P` before trusting it.
- Checks whether the **entire** history is consistent with repeating `history[-P:]`.
- If so, predicts the next move in the cycle and plays its counter.

Catches: repeaters (period 1), alternators (period 2), classic R→P→S cyclers (period 3).

### 2. Bigram N-gram (N=2)

Builds a `{(prev2, prev1): Counter(next)}` table from the full history. Fires only if the most-common follower has count ≥ 2 and is strictly ahead of second place.

### 3. Unigram N-gram (N=1)

Same as bigram but keyed on just the last move. Catches simple "if I played X they play Y" reactive patterns.

### 4. Frequency Bias

If any single move appears ≥ 40% of the time (minimum 5 games), play its counter.

### 5. Default

Always play `paper`.

## Assumptions

- `history` contains only the **opponent's** past moves (not the bot's).
- The sliding window in `test_sim.py` is capped at 10 moves (configurable), matching a typical real-time game constraint.
- Opponents use fixed strategies; the bot adapts within a 200-game session.

## Trade-offs

| Decision             | Rationale                                                                                         |
| -------------------- | ------------------------------------------------------------------------------------------------- |
| Period cap at 3      | Requires only 2*P history entries; longer periods need more data than a 200-game session provides |
| No external ML       | Keeps the solution dependency-free and well under the 100 ms time limit                           |
| No state persistence | Each invocation is stateless; history is passed in via stdin each round                           |
| Default = paper      | Exploits rock bias common in humans and naive bots                                                |

## Expected Win/Tie Rates

| Opponent type | Strategy that fires            | Expected rate |
| ------------- | ------------------------------ | ------------- |
| Repeater      | Cycle period 1 (after 2 moves) | ~90–95%       |
| R→P→S cycler  | Cycle period 3 (after 6 moves) | ~85–90%       |
| Alternator    | Cycle period 2 (after 4 moves) | ~85–90%       |
| Reactive      | Bigram / unigram N-gram        | ~70–80%       |
| Pure random   | Default (paper)                | ~33–40%       |

Random opponents are the weak point; heavy crushing of patterned opponents pulls the overall average well above 70%.
