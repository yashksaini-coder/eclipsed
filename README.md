# Eclipsed?

A Kaggle benchmark for whether a model can tell if a Kademlia lookup is under an eclipse attack, built so the obvious clue (IP clustering) and the real clue (ID distribution) disagree.

Entry for the [DEV x Kaggle Benchmarking Challenge](https://dev.to/devteam/join-the-kaggle-benchmarking-challenge-2500-in-prizes-for-five-winners-18ml). Kaggle benchmark: _link added once published_.

## Why this benchmark

Every eclipse-attack explainer says "look for many peers from one subnet". That is the shortcut. The real signal is that Sybil IDs sit implausibly close to the key compared with what `n` uniformly random honest IDs would produce. The verdict task is a 2x2 where those two signals are decoupled, so a model that only reads the IP column scores at chance on half the cells. Every item comes from a seeded generator with exact ground truth, so nothing can have been memorised.

## Layout

```
eclipsed/            library, standard library only so it can be inlined into a Kaggle cell
  gen.py             seeded item generator: xor, bucket, verdict (+ verdict_sweep), admission
  baseline.py        K-L divergence detector and the IP-only shortcut, as reference points
tasks/               Kaggle task sources, one file per task; they call the generator by name
dist/                built single-file tasks: the thing you paste into a Kaggle benchmark notebook
tests/               ground truth re-derived by an independent route, plus the built task run
                     end to end against scripted models
build.py             inlines eclipsed/gen.py into each task source and writes dist/
```

## Use

```bash
uv sync                                   # Python 3.12, kaggle-benchmarks, pytest
uv run python -m eclipsed.gen --out items # dump every family as JSONL
uv run python -m eclipsed.baseline        # reference detectors on the 2x2 and the stealth sweep
uv run python build.py                    # rebuild dist/ after editing sources
uv run pytest                             # ~1 min
```

On Kaggle: open https://www.kaggle.com/benchmarks/tasks/new, paste `dist/task1_xor_distance.py`, uncomment the last line (`%choose xor_distance`), save a version, then add models from the task page.

## Tasks

| # | family      | question                                                        | trap                                                      |
|---|-------------|-----------------------------------------------------------------|-----------------------------------------------------------|
| 1 | `xor`       | which of 8 peers is XOR-closest to the key                      | half the items have a peer that is closer by subtraction  |
| 2 | `bucket`    | which k-bucket (common prefix length) a peer belongs in         | numerically adjacent IDs that share a short prefix        |
| 3 | `verdict`   | is this closest-peer set honest or eclipsed                     | the 2x2 below                                             |
| 4 | `admission` | admit or reject a peer under a subnet-diversity limit           | IPv6 grouped by /48, with /64 look-alikes                 |

Task 1 is built and tested in `dist/`. The other families generate and verify but have no Kaggle task file yet.

## Verdict design

Each item is the 20 closest peers a lookup returned, with the network size and the share of honest nodes at one hosting provider stated in the prompt.

|                     | provider share 50% | provider share 5% |
|---------------------|--------------------|-------------------|
| Sybil IDs inserted  | obvious attack     | stealth attack    |
| honest IDs only     | red herring        | clean             |

- The count of listed peers inside the provider /16 is drawn from the stated share whether or not the set is attacked, so the IP column alone carries no label. `tests/test_items.py::test_ip_column_carries_no_label` checks that the best IP-only lookup table scores within 4 points of a coin flip.
- Decoy gap = accuracy on (obvious attack, clean) minus accuracy on (stealth attack, red herring).

Reference points on 400 items:

| detector      | obvious | stealth | red herring | clean | decoy gap |
|---------------|---------|---------|-------------|-------|-----------|
| K-L baseline  | 1.00    | 1.00    | 0.98        | 1.00  | 0.01      |
| IP heuristic  | 1.00    | 0.02    | 0.01        | 1.00  | 0.98      |

`uv run python -m eclipsed.baseline` also prints the baseline's detection rate across attacker stealth (grind bits by Sybil count).

## License

MIT.
