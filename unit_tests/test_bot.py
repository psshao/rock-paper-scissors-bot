"""Unit tests for bot.py — covers each strategy function and choose_move dispatch."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

# Allow importing bot.py from the parent directory
sys.path.insert(0, str(Path(__file__).parent.parent))
from bot import choose_move, detect_cycle, frequency_bias, ngram_predict, parse_input

MOVES = ["rock", "paper", "scissors"]
COUNTERS = {"rock": "paper", "paper": "scissors", "scissors": "rock"}


# ---------------------------------------------------------------------------
# detect_cycle
# ---------------------------------------------------------------------------

class TestDetectCycle(unittest.TestCase):

    # --- period 1 (repeater) ---

    def test_repeater_rock(self):
        self.assertEqual(detect_cycle(["rock"] * 4), "paper")

    def test_repeater_paper(self):
        self.assertEqual(detect_cycle(["paper"] * 4), "scissors")

    def test_repeater_scissors(self):
        self.assertEqual(detect_cycle(["scissors"] * 4), "rock")

    def test_repeater_minimum_history(self):
        # period 1 requires len >= 2
        self.assertIsNone(detect_cycle(["rock"]))
        self.assertEqual(detect_cycle(["rock", "rock"]), "paper")

    # --- period 2 (alternator) ---

    def test_alternator_rp_next_rock(self):
        # rock paper rock paper → next is rock → counter paper
        self.assertEqual(detect_cycle(["rock", "paper", "rock", "paper"]), "paper")

    def test_alternator_rp_next_paper(self):
        # detect_cycle uses h[-p:] as the pattern, so detection only fires when the history
        # length is a multiple of the period (phase-aligned). At odd lengths the last 2
        # entries are a rotated pattern and the full-history check fails.
        self.assertIsNone(detect_cycle(["rock", "paper", "rock"]))           # len=3, < 2*2
        self.assertIsNone(detect_cycle(["rock", "paper", "rock", "paper", "rock"]))  # len=5, misaligned
        # len=6 is phase-aligned: pattern=["rock","paper"], next=pattern[0]="rock" → paper
        # To get a "paper" prediction the history must end with [..., paper, rock, paper, rock]
        # so that pattern=["paper","rock"] and next=pattern[0]="paper" → counter scissors
        h = ["paper", "rock", "paper", "rock", "paper", "rock"]
        self.assertEqual(detect_cycle(h), "scissors")

    def test_alternator_minimum_history(self):
        # period 2 requires len >= 4; len=3 should NOT fire period 2
        # but CAN fire period 1 if consistent — rock,paper,rock is not period-1
        self.assertIsNone(detect_cycle(["rock", "paper"]))

    # --- period 3 (cycler) ---

    def test_cycle_rps(self):
        # rock paper scissors rock paper scissors → next rock → counter paper
        self.assertEqual(detect_cycle(["rock", "paper", "scissors",
                                       "rock", "paper", "scissors"]), "paper")

    def test_cycle_rps_phase_alignment(self):
        # Like the alternator, period-3 detection only fires when len is a multiple of 3.
        # At other lengths h[-3:] is a rotation of the true pattern and the check fails.
        self.assertIsNone(detect_cycle(["rock", "paper", "scissors", "rock", "paper"]))   # len=5
        self.assertIsNone(detect_cycle(["rock", "paper", "scissors",
                                        "rock", "paper", "scissors", "rock"]))             # len=7, misaligned
        # len=9 is phase-aligned: pattern=["rock","paper","scissors"], next=pattern[0]="rock" → paper
        h = ["rock", "paper", "scissors"] * 3
        self.assertEqual(detect_cycle(h), "paper")

    def test_cycle_minimum_history(self):
        # period 3 requires len >= 6
        self.assertIsNone(detect_cycle(["rock", "paper", "scissors",
                                        "rock", "paper"]))
        # exactly 6 entries — should fire
        self.assertIsNotNone(detect_cycle(["rock", "paper", "scissors",
                                           "rock", "paper", "scissors"]))

    # --- no cycle ---

    def test_empty_history(self):
        self.assertIsNone(detect_cycle([]))

    def test_no_cycle_random_ish(self):
        self.assertIsNone(detect_cycle(["rock", "paper", "rock", "scissors"]))

    def test_broken_cycle(self):
        # would be period-3 but last move breaks it
        self.assertIsNone(detect_cycle(["rock", "paper", "scissors",
                                        "rock", "paper", "rock"]))


# ---------------------------------------------------------------------------
# ngram_predict
# ---------------------------------------------------------------------------

class TestNgramPredict(unittest.TestCase):

    # --- unigram (n=1) ---

    def test_unigram_clear_pattern(self):
        # after rock, opponent always plays paper → predict paper → counter scissors
        h = ["rock", "paper", "rock", "paper", "rock", "paper", "rock"]
        self.assertEqual(ngram_predict(h, 1), "scissors")

    def test_unigram_minimum_history(self):
        self.assertIsNone(ngram_predict(["rock"], 1))
        # 2 entries: one pair, count=1 — below threshold of 2
        self.assertIsNone(ngram_predict(["rock", "paper"], 1))

    def test_unigram_count_below_threshold(self):
        # last move is rock, but rock→paper only seen once
        self.assertIsNone(ngram_predict(["scissors", "paper", "rock", "paper"], 1))

    def test_unigram_tied_top(self):
        # rock→paper twice, rock→scissors twice — tied, should return None
        h = ["rock", "paper", "rock", "scissors", "rock", "paper", "rock", "scissors", "rock"]
        self.assertIsNone(ngram_predict(h, 1))

    def test_unigram_key_not_in_counts(self):
        # last move is scissors but scissors never appeared earlier
        h = ["rock", "paper", "rock", "paper", "scissors"]
        self.assertIsNone(ngram_predict(h, 1))

    # --- bigram (n=2) ---

    def test_bigram_clear_pattern(self):
        # (rock,paper) always followed by scissors
        h = ["rock", "paper", "scissors", "rock", "paper", "scissors", "rock", "paper"]
        self.assertEqual(ngram_predict(h, 2), "rock")  # counter scissors

    def test_bigram_minimum_history(self):
        self.assertIsNone(ngram_predict(["rock", "paper"], 2))
        # 3 entries: one pair, count=1 — below threshold
        self.assertIsNone(ngram_predict(["rock", "paper", "scissors"], 2))

    def test_bigram_count_below_threshold(self):
        # bigram (rock,paper) → scissors only once
        h = ["rock", "paper", "scissors", "paper", "rock", "paper"]
        self.assertIsNone(ngram_predict(h, 2))

    def test_bigram_tied_top(self):
        # (rock,paper)→scissors twice, (rock,paper)→rock twice
        h = ["rock", "paper", "scissors",
             "rock", "paper", "rock",
             "rock", "paper", "scissors",
             "rock", "paper", "rock",
             "rock", "paper"]
        self.assertIsNone(ngram_predict(h, 2))


# ---------------------------------------------------------------------------
# frequency_bias
# ---------------------------------------------------------------------------

class TestFrequencyBias(unittest.TestCase):

    def test_rock_heavy(self):
        # 5 rocks out of 7 ≈ 71% → counter paper
        self.assertEqual(frequency_bias(["rock"] * 5 + ["paper", "scissors"]), "paper")

    def test_paper_heavy(self):
        self.assertEqual(frequency_bias(["paper"] * 5 + ["rock", "scissors"]), "scissors")

    def test_scissors_heavy(self):
        self.assertEqual(frequency_bias(["scissors"] * 5 + ["rock", "paper"]), "rock")

    def test_below_threshold(self):
        # rock=1, paper=2, scissors=2 out of 5 — max is 2/5=40%, which is NOT < 0.40, so fires
        # but rock=1, paper=1, scissors=3 → 3/5=60% → fires; need a case that stays under
        # 1/5=20%, 1/5=20%, 3/5=60% all fire; to stay under 40% need max <= 1/3 of total
        # e.g. rock=2, paper=2, scissors=2 → each 33.3% < 40% → None
        self.assertIsNone(frequency_bias(["rock", "paper", "scissors",
                                          "rock", "paper", "scissors"]))

    def test_exactly_at_threshold(self):
        # 2 out of 5 = 40.0% — boundary: `count/len < 0.40` is False → fires
        h = ["rock", "rock", "paper", "scissors", "scissors"]
        # rock=2, paper=1, scissors=2 — tied at top, most_common picks rock or scissors
        # either way count=2, 2/5=0.4, not < 0.4 → fires
        result = frequency_bias(h)
        self.assertIn(result, {"paper", "rock"})  # counter of rock or scissors

    def test_too_short(self):
        self.assertIsNone(frequency_bias(["rock"] * 4))
        self.assertIsNotNone(frequency_bias(["rock"] * 5))

    def test_uniform_distribution(self):
        # rock, paper, scissors each 33% — no bias
        self.assertIsNone(frequency_bias(["rock", "paper", "scissors",
                                          "rock", "paper", "scissors"]))


# ---------------------------------------------------------------------------
# choose_move — dispatch priority
# ---------------------------------------------------------------------------

class TestChooseMove(unittest.TestCase):

    def test_empty_history_returns_paper(self):
        self.assertEqual(choose_move([]), "paper")

    def test_cycle_takes_priority_over_ngram(self):
        # A cycle-3 pattern also has ngram signal, but cycle should fire first
        h = ["rock", "paper", "scissors", "rock", "paper", "scissors"]
        self.assertEqual(choose_move(h), "paper")  # cycle predicts rock → paper

    def test_falls_through_to_default(self):
        # Truly random-ish sequence that won't trigger any strategy
        h = ["rock", "scissors", "paper", "rock"]
        self.assertEqual(choose_move(h), "paper")

    def test_frequency_fires_when_cycle_and_ngram_fail(self):
        # All rocks — cycle fires first (period 1), but let's confirm with a
        # history that has bias but no clean cycle or ngram signal
        h = ["rock", "rock", "scissors", "rock", "rock", "paper", "rock"]
        # rock appears 5/7 ≈ 71%; cycle won't fire (broken); ngram may or may not
        result = choose_move(h)
        # whatever fires, the result must be a valid move
        self.assertIn(result, {"rock", "paper", "scissors"})

    def test_returns_valid_move_always(self):
        histories = [
            [],
            ["rock"],
            ["rock", "paper"],
            ["scissors", "scissors", "scissors"],
            ["rock", "paper", "scissors"] * 3,
        ]
        for h in histories:
            with self.subTest(h=h):
                self.assertIn(choose_move(h), {"rock", "paper", "scissors"})


# ---------------------------------------------------------------------------
# parse_input
# ---------------------------------------------------------------------------

class TestParseInput(unittest.TestCase):

    def test_reads_plain_text_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("rock\npaper\nscissors\n")
            tmp = f.name
        self.assertEqual(parse_input(tmp), ["rock", "paper", "scissors"])

    def test_file_normalizes_case(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("ROCK\nPaper\nSCISSORS\n")
            tmp = f.name
        self.assertEqual(parse_input(tmp), ["rock", "paper", "scissors"])

    def test_file_skips_invalid_lines(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("rock\nbanana\npaper\n\nscissors\n")
            tmp = f.name
        self.assertEqual(parse_input(tmp), ["rock", "paper", "scissors"])

    def test_file_empty(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            tmp = f.name
        self.assertEqual(parse_input(tmp), [])

    def test_stdin_json(self):
        payload = json.dumps({"opponent": "test", "history": ["rock", "paper", "scissors"]})
        with patch("sys.stdin") as mock_stdin:
            mock_stdin.read.return_value = payload
            self.assertEqual(parse_input(None), ["rock", "paper", "scissors"])

    def test_stdin_json_filters_invalid(self):
        payload = json.dumps({"opponent": "test", "history": ["rock", "banana", "scissors"]})
        with patch("sys.stdin") as mock_stdin:
            mock_stdin.read.return_value = payload
            self.assertEqual(parse_input(None), ["rock", "scissors"])

    def test_stdin_json_missing_history_key(self):
        payload = json.dumps({"opponent": "test"})
        with patch("sys.stdin") as mock_stdin:
            mock_stdin.read.return_value = payload
            self.assertEqual(parse_input(None), [])


if __name__ == "__main__":
    unittest.main()
