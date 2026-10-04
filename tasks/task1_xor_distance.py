# %% [markdown]
# # Eclipsed? Task 1: XOR distance
#
# Pick the peer closest to a key under Kademlia's XOR metric. Half the items
# are traps: the peer that is closest as an ordinary number is not the
# XOR-closest one. Items come from a seeded generator, so ground truth is
# exact and nothing here can have been memorised.

# %%
import math
import re
import time

import pandas as pd

import kaggle_benchmarks as kbench

SEED = 2026
N_ITEMS = 60  # half 'agree', half 'trap'

df = pd.DataFrame(xor_items(N_ITEMS, seed=SEED))[  # noqa: F821 (inlined above)
    ["item_id", "kind", "prompt", "answer", "decoy"]
]

ANSWER_RE = re.compile(r"ANSWER:[\s*`]*(?:0x)?([0-9a-fA-F]{8})\b")


def parse_answer(text: str) -> str | None:
    found = ANSWER_RE.findall(text or "")
    return found[-1].lower() if found else None


# %%
@kbench.task(name="xor_closest_peer_item", store_task=False)
def xor_item(llm, item_id, kind, prompt, answer, decoy) -> dict:
    picked = parse_answer(str(llm.prompt(prompt)))
    return {
        "item_id": item_id,
        "kind": kind,
        "picked": picked,
        "parsed": picked is not None,
        "correct": picked == answer,
        "took_decoy": kind == "trap" and picked == decoy,
    }


def run_items(llm, attempts: int = 3) -> pd.DataFrame:
    """One row per item. Items that still error after every attempt count as wrong.

    The retry loop lives here because the SDK forces max_attempts=1 on an
    evaluate() nested inside another task.
    """
    done, todo = {}, df
    for attempt in range(attempts):
        with kbench.client.enable_cache():
            runs = xor_item.evaluate(
                llm=[llm],
                evaluation_data=todo,
                n_jobs=4,
                timeout=300,
                on_failure="continue",
                remove_run_files=True,
            )
        done |= {r.result["item_id"]: r.result for r in runs.completed_runs}
        todo = df[~df.item_id.isin(done.keys())]
        if todo.empty or attempt == attempts - 1:
            break
        time.sleep(5)
    blank = {"picked": None, "parsed": False, "correct": False, "took_decoy": False}
    out = pd.DataFrame(
        [
            {"item_id": i, "kind": k, "errored": i not in done} | done.get(i, blank)
            for i, k in zip(df.item_id, df.kind)
        ]
    )
    return out


def summarise(res: pd.DataFrame) -> dict:
    trap, agree = res[res.kind == "trap"], res[res.kind == "agree"]
    return {
        "accuracy": res.correct.mean(),
        "agree_accuracy": agree.correct.mean(),
        "trap_accuracy": trap.correct.mean(),
        "decoy_gap": agree.correct.mean() - trap.correct.mean(),
        "took_numeric_decoy": trap.took_decoy.mean(),
        "unparsed": (~res.parsed & ~res.errored).mean(),
        "errored": int(res.errored.sum()),
    }


# %%
@kbench.task(
    name="Eclipsed 1: XOR distance",
    description="Pick the XOR-closest peer to a key; half the items have a numerically closer decoy.",
)
def xor_distance(llm) -> tuple[float, float]:
    res = run_items(llm)
    stats = summarise(res)
    print({k: round(float(v), 3) for k, v in stats.items()})
    p, n = float(stats["accuracy"]), len(res)
    return p, 1.96 * math.sqrt(p * (1 - p) / n)


run = xor_distance.run(kbench.llm)
run

# %% [markdown]
# ## Side-by-side breakdown (optional)
# The leaderboard score is overall accuracy. For the write-up, list model
# names in `COMPARE` to get the agree / trap split per model in one table.
# `sorted(kbench.llms)` shows what your account can run.

# %%
COMPARE: list[str] = []  # e.g. ["google/gemini-2.5-flash", "meta/llama-3.1-70b"]

if COMPARE:
    table = pd.DataFrame(
        {name: summarise(run_items(kbench.llms[name])) for name in COMPARE}
    ).T
    print(table.round(3).to_string())

# %%
# %choose xor_distance
