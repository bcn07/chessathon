"""The pair (pentanomial) statistics behind every A/B verdict."""

from bench.stats import SPRT_BOUND, elo_from_score, llr, pair_scores, summarise, verdict


def test_pairs_follow_game_index_not_completion_order() -> None:
    # games 0/1 are one opening pair, 2/3 another; recorded out of order (0, 2, 1, 3):
    # pairing by index gives two level pairs, pairing by position would give 1.0 and 0.0
    assert pair_scores([1.0, 1.0, 0.0, 0.0], [0, 2, 1, 3]) == [0.5, 0.5]
    assert pair_scores([1.0, 0.5, 0.0], None) == [0.75, 0.0]  # lone last game counts alone


def test_pair_interval_is_tight_when_colours_cancel() -> None:
    # win as white, lose as black in every pair: per game this looks like 50% ± a lot,
    # per pair it is exactly 50% with zero variance
    results = [1.0, 0.0] * 50
    assert "+0 ± 0 Elo" in summarise(results, list(range(100)))


def test_elo_and_llr_signs() -> None:
    assert elo_from_score(0.5) == 0.0
    assert elo_from_score(0.75) > 190
    strong = [1.0, 0.5] * 100  # 75%
    weak = [0.0, 0.5] * 100
    assert llr(strong, None, 0, 10) > SPRT_BOUND
    assert llr(weak, None, 0, 10) < -SPRT_BOUND
    assert verdict(SPRT_BOUND, 0, 10) == "H1 accepted: at least +10 Elo"
    assert verdict(-SPRT_BOUND, 0, 10) == "H0 accepted: no better than +0 Elo"
    assert verdict(0.0, 0, 10) is None


def test_lopsided_results_still_decide() -> None:
    assert llr([1.0] * 40, None, 0, 10) > SPRT_BOUND
