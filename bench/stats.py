"""Match statistics shared by the gauntlet and the aggregate.

Games come in pairs: the same opening with the colours swapped. Scoring by pairs (the
"pentanomial" model) counts the correlation between the two games of a pair, which the per-game
model ignores, so its interval is the honest one; ``pair_scores`` falls back to single games
when a pair is incomplete. The sequential test is fishtest's GSPRT approximation, written on
the pair scores.
"""

from __future__ import annotations

import math

SPRT_BOUND = math.log(19)  # alpha = beta = 0.05


def elo_from_score(score: float) -> float:
    score = min(max(score, 1e-6), 1 - 1e-6)
    return -400.0 * math.log10(1.0 / score - 1.0) + 0.0  # + 0.0 turns -0.0 into 0.0


def expected_score(elo: float) -> float:
    return 1 / (1 + 10 ** (-elo / 400))


def pair_scores(results: list[float], order: list[int] | None = None) -> list[float]:
    """Mean score of each opening pair (games 2k and 2k+1); lone games count on their own."""
    if order is None or len(order) != len(results):
        order = list(range(len(results)))
    halves: dict[int, list[float]] = {}
    for game, points in zip(order, results, strict=True):
        halves.setdefault(game // 2, []).append(points)
    return [sum(v) / len(v) for v in halves.values()]


def mean_and_se(scores: list[float]) -> tuple[float, float]:
    n = len(scores)
    if n == 0:
        return 0.5, float("inf")
    mean = sum(scores) / n
    variance = sum((s - mean) ** 2 for s in scores) / max(n - 1, 1)
    return mean, math.sqrt(variance / n)


def summarise(results: list[float], order: list[int] | None = None) -> str:
    """Score with an Elo difference and a 95% interval from the pair (pentanomial) variance."""
    if not results:
        return "no games"
    pairs = pair_scores(results, order)
    mean, se = mean_and_se(pairs)
    low, high = mean - 1.96 * se, mean + 1.96 * se
    elo = elo_from_score(mean)
    if low > 0 and high < 1:
        spread = (elo_from_score(high) - elo_from_score(low)) / 2
        interval = f"{elo:+.0f} ± {spread:.0f} Elo"
    else:
        interval = f"{elo:+.0f} Elo (interval unbounded at this sample size)"
    return f"score {mean:.1%}, {interval} ({len(pairs)} pairs)"


def llr(results: list[float], order: list[int] | None, elo0: float, elo1: float) -> float:
    """GSPRT log-likelihood ratio of H1 (elo1) over H0 (elo0) from the pair scores."""
    pairs = pair_scores(results, order)
    n = len(pairs)
    if n < 2:
        return 0.0
    mean, se = mean_and_se(pairs)
    variance = se * se * n
    if variance <= 0:
        # every pair identical: nudge so a lopsided result still reaches a verdict
        variance = 0.25 / (n + 1)
    s0, s1 = expected_score(elo0), expected_score(elo1)
    return (s1 - s0) * (2 * mean - s0 - s1) / (2 * variance / n)


def sprt_llr(wins: int, draws: int, losses: int, elo0: float, elo1: float) -> float:
    """Per-game (trinomial) GSPRT approximation, kept for callers without pair information."""
    n = wins + draws + losses
    if n < 2:
        return 0.0
    total = n + 1.5
    w, d = (wins + 0.5) / total, (draws + 0.5) / total
    mean = w + d / 2
    variance = (w + d / 4) - mean**2
    if variance <= 0:
        return 0.0
    s0, s1 = expected_score(elo0), expected_score(elo1)
    return (s1 - s0) * (2 * mean - s0 - s1) / (2 * variance / total)


def verdict(value: float, elo0: float, elo1: float) -> str | None:
    if value >= SPRT_BOUND:
        return f"H1 accepted: at least {elo1:+g} Elo"
    if value <= -SPRT_BOUND:
        return f"H0 accepted: no better than {elo0:+g} Elo"
    return None
