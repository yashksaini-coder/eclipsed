"""Reference detectors for the verdict task. Standard library only.

kl_detector   -- K-L divergence between the observed common-prefix-length
                 histogram and the one an honest network of the stated size
                 would produce. Threshold = 99th percentile of the statistic
                 on simulated honest sets, so it is never tuned on attacks.
ip_heuristic  -- "ECLIPSED if five or more peers share a /16". The shortcut
                 the 2x2 design is built to expose.
"""

import functools
import math
import random
from collections import Counter

from kadbench.gen import BITS, K, honest_closest, verdict_items, verdict_sweep

CONFLICT = ("stealth_attack", "red_herring")
AGREE = ("obvious_attack", "clean")


@functools.lru_cache(maxsize=None)
def expected_cpl_pmf(n: int, k: int = K) -> tuple:
    """P(cpl = c) for a peer drawn from the k closest of n uniform IDs."""

    def at_least(c):  # P(cpl >= c): mean over ranks i of P(Gamma(i) < n / 2^c)
        x = n / 2**c
        term, cdf, total = math.exp(-x), 0.0, 0.0
        for i in range(1, k + 1):
            cdf += term  # e^-x * sum_{j<i} x^j / j!
            total += 1.0 - cdf
            term *= x / i
        return total / k

    tail = [at_least(c) for c in range(BITS + 2)]
    return tuple(max(tail[c] - tail[c + 1], 0.0) for c in range(BITS + 1))


def kl_statistic(cpls, n: int) -> float:
    expected = expected_cpl_pmf(n)
    counts = Counter(cpls)
    total = len(cpls)
    return sum(
        (m / total) * math.log((m / total) / max(expected[c], 1e-9))
        for c, m in counts.items()
    )


@functools.lru_cache(maxsize=None)
def kl_threshold(n: int, trials: int = 4000, quantile: float = 0.99) -> float:
    rng = random.Random(f"kl-threshold-{n}")
    stats = sorted(
        kl_statistic([BITS - d.bit_length() for d in honest_closest(rng, n)], n)
        for _ in range(trials)
    )
    return stats[int(quantile * trials)]


def kl_detector(item) -> str:
    stat = kl_statistic([p["cpl"] for p in item["peers"]], item["n"])
    return "ECLIPSED" if stat > kl_threshold(item["n"]) else "HEALTHY"


def ip_heuristic(item) -> str:
    counts = Counter(p["ip"].rsplit(".", 2)[0] for p in item["peers"])
    return "ECLIPSED" if max(counts.values()) >= 5 else "HEALTHY"


def score(items, detector) -> dict:
    """Per-cell accuracy plus the decoy gap (agree cells minus conflict cells)."""
    hits, totals = Counter(), Counter()
    for item in items:
        totals[item["cell"]] += 1
        hits[item["cell"]] += detector(item) == item["label"]
    acc = {cell: hits[cell] / totals[cell] for cell in totals}
    mean = lambda cells: sum(acc[c] for c in cells) / len(cells)
    acc["decoy_gap"] = mean(AGREE) - mean(CONFLICT)
    return acc


def main():
    items = verdict_items(n=400)
    print("2x2 core set, 100 items per cell\n")
    cols = ("obvious_attack", "stealth_attack", "red_herring", "clean", "decoy_gap")
    print(f"{'detector':14s}" + "".join(f"{c:>16s}" for c in cols))
    for name, det in (("K-L baseline", kl_detector), ("IP heuristic", ip_heuristic)):
        acc = score(items, det)
        print(f"{name:14s}" + "".join(f"{acc[c]:16.2f}" for c in cols))

    print("\nStealth sweep: K-L baseline detection rate (40 attacked sets per point)")
    sweep = verdict_sweep(per_point=40)
    grinds = sorted({i["grind"] for i in sweep})
    counts = sorted({i["sybils_inserted"] for i in sweep})
    print("grind bits \\ Sybils" + "".join(f"{s:>7d}" for s in counts))
    for g in grinds:
        row = []
        for s in counts:
            pt = [i for i in sweep if i["grind"] == g and i["sybils_inserted"] == s]
            row.append(sum(kl_detector(i) == "ECLIPSED" for i in pt) / len(pt))
        print(f"{g:>18d}" + "".join(f"{r:7.2f}" for r in row))


if __name__ == "__main__":
    main()
