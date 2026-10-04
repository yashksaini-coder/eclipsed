# Built by build.py from eclipsed/gen.py + tasks/task1_xor_distance.py. Edit the sources, not this file.

# %%
# Item generator (standard library only)
"""Eclipsed? -- seeded item generator for the Kademlia eclipse benchmark.

Standard library only, so the whole module can be inlined into a Kaggle task.
Every item carries its own prompt and exact ground truth; nothing is judged.

Families
  xor        pick the XOR-closest peer            (task 1)
  bucket     common-prefix-length bucket index    (task 2)
  verdict    healthy vs eclipsed, 2x2 design      (tasks 3 and 5)
  admission  subnet-limit admit / reject          (task 4)
"""

import ipaddress
import json
import random

BITS = 32
SPACE = 1 << BITS
K = 20


def hx(x: int) -> str:
    return f"{x:08x}"


def cpl(a: int, b: int) -> int:
    """Common prefix length in bits (the libp2p bucket index)."""
    d = a ^ b
    return BITS if d == 0 else BITS - d.bit_length()


# --------------------------------------------------------------------------
# Task 1: XOR distance
# --------------------------------------------------------------------------

XOR_PROMPT = """\
Kademlia measures the distance between two IDs as their bitwise XOR, read as an unsigned integer. IDs here are 32 bits, written as 8 hex digits.

Target key: {target}

Peers:
{peers}

Which peer is closest to the target key under the XOR metric?

You may show your working. Finish with one final line in exactly this form:
ANSWER: <peer id>"""


def _boundary_pair(rng, b, slack):
    """Two IDs numerically adjacent across a 2^b boundary, so XOR-far."""
    hi = rng.getrandbits(BITS - 1 - b) << (b + 1)
    r1 = rng.randrange(slack)
    r2 = rng.randrange(slack)
    above = hi | (1 << b) | r1
    below = hi | ((1 << b) - 1 - r2)
    return (above, below) if rng.random() < 0.5 else (below, above)


def gen_xor_item(rng: random.Random, kind: str, n_peers: int = 8) -> dict:
    """kind='trap': the numerically closest peer is NOT the XOR-closest.
    kind='agree': both notions pick the same peer."""
    while True:
        b = rng.randint(10, 30)
        target, decoy = _boundary_pair(rng, b, 1 << (b - 6))
        closest = target ^ rng.randrange(1 << (b - 3), 1 << (b - 1))
        peers = {closest}
        if kind == "trap":
            peers.add(decoy)
        while len(peers) < n_peers:
            y = rng.getrandbits(rng.randint(b + 1, BITS)) | (1 << b)
            p = target ^ y
            if p != target:
                peers.add(p)
        peers = list(peers)
        rng.shuffle(peers)
        by_xor = min(peers, key=lambda p: p ^ target)
        num = sorted(abs(p - target) for p in peers)
        if num[0] == num[1]:
            continue
        by_num = min(peers, key=lambda p: abs(p - target))
        if (by_xor != by_num) == (kind == "trap"):
            break
    return {
        "kind": kind,
        "target": hx(target),
        "peers": ",".join(hx(p) for p in peers),
        "answer": hx(by_xor),
        "decoy": hx(by_num),  # equals answer on 'agree' items
        "prompt": XOR_PROMPT.format(
            target=hx(target), peers="\n".join(hx(p) for p in peers)
        ),
    }


def xor_items(n: int = 60, seed: int = 2026) -> list[dict]:
    rng = random.Random(f"xor-{seed}")
    out = []
    for i in range(n):
        item = gen_xor_item(rng, "trap" if i % 2 else "agree")
        item["item_id"] = f"xor-{i:03d}"
        out.append(item)
    return out


# --------------------------------------------------------------------------
# Task 2: bucket placement
# --------------------------------------------------------------------------

BUCKET_PROMPT = """\
A Kademlia node places each peer in the bucket whose index is the common prefix length: the number of leading bits the peer's ID shares with the node's own ID. Bucket 0 means the very first bit differs. IDs are 32 bits, written as 8 hex digits.

Local node ID: {local}
Peer ID:       {peer}

Which bucket index does this peer belong in?

You may show your working. Finish with one final line in exactly this form:
ANSWER: <integer>"""


def gen_bucket_item(rng: random.Random, kind: str) -> dict:
    """kind='boundary': IDs are numerically adjacent but share a short prefix."""
    if kind == "boundary":
        b = rng.randint(8, 28)
        local, peer = _boundary_pair(rng, b, 100)
    else:
        c = rng.randint(0, 24)
        local = rng.getrandbits(BITS)
        low = BITS - 1 - c
        peer = local ^ ((1 << low) | rng.getrandbits(low))
    return {
        "kind": kind,
        "local": hx(local),
        "peer": hx(peer),
        "answer": cpl(local, peer),
        "prompt": BUCKET_PROMPT.format(local=hx(local), peer=hx(peer)),
    }


def bucket_items(n: int = 60, seed: int = 2026) -> list[dict]:
    rng = random.Random(f"bucket-{seed}")
    out = []
    for i in range(n):
        item = gen_bucket_item(rng, "boundary" if i % 2 else "uniform")
        item["item_id"] = f"bucket-{i:03d}"
        out.append(item)
    return out


# --------------------------------------------------------------------------
# Task 3: verdict (the 2x2)
# --------------------------------------------------------------------------

VERDICT_PROMPT = """\
You are auditing a Kademlia DHT node. IDs are 32 bits (8 hex digits), distance is XOR, and lookups return the k = 20 closest peers.

The node looked up key {target}. The 20 peers it got back are listed below.

Network facts:
- About {n} honest nodes are online, with IDs spread uniformly at random.
- About {h_pct}% of honest nodes are hosted at one provider, inside {prefix}.0.0/16. The rest are spread across unrelated networks.

{header}
{rows}
{legend}
Is this result set what an honest network would produce, or has an attacker inserted Sybil peers whose IDs were generated to sit close to the key (an eclipse attack)?

You may show your reasoning. Finish with exactly these two lines:
VERDICT: HEALTHY or ECLIPSED
SUSPECTS: comma-separated peer ids you believe are Sybils, or NONE"""

CELLS = {
    (True, True): "obvious_attack",
    (True, False): "stealth_attack",
    (False, True): "red_herring",
    (False, False): "clean",
}
NETWORK_SIZES = (1000, 4000, 16000, 64000)


def honest_closest(rng: random.Random, n: int, k: int = K) -> list[int]:
    """XOR distances of the k closest of n uniform IDs (exact order statistics)."""
    while True:
        acc, gaps = 0.0, []
        for _ in range(k):
            acc += rng.expovariate(1.0)
            gaps.append(acc)
        total = acc + rng.gammavariate(n + 1 - k, 1.0)
        d = [int(SPACE * g / total) for g in gaps]
        if d[0] > 0 and len(set(d)) == k:
            return d


def _slash16(rng, taken):
    while True:
        a, b = rng.randint(1, 223), rng.randint(0, 255)
        private = (
            a in (10, 127)
            or (a == 172 and 16 <= b <= 31)
            or (a == 192 and b == 168)
            or (a == 169 and b == 254)
            or (a == 100 and 64 <= b <= 127)
        )
        if not private and (a, b) not in taken:
            taken.add((a, b))
            return a, b


def gen_verdict_item(
    rng: random.Random,
    attacked: bool,
    clustered: bool,
    n: int | None = None,
    sybils: int | None = None,
    grind: int | None = None,
    show_cpl: bool = True,
) -> dict:
    """One closest-peer set.

    attacked  -- `sybils` peers are inserted uniformly inside a radius 2^grind
                 times smaller than the honest 20th-closest distance.
    clustered -- the stated share of honest nodes at one provider is 50%
                 (else 5%). The number of listed peers inside the provider /16
                 is drawn from that share whether or not the set is attacked,
                 so the IP column alone says nothing about the label. When
                 attacked, provider addresses go to Sybils first.
    """
    n = n or rng.choice(NETWORK_SIZES)
    sybils = sybils if sybils is not None else rng.randint(12, 16)
    grind = grind if grind is not None else rng.randint(4, 6)
    target = rng.getrandbits(BITS)

    dist = {d: False for d in honest_closest(rng, n)}
    if attacked:
        radius = max(sybils + 2, int(SPACE * K / n) >> grind)
        added = 0
        while added < sybils:
            d = rng.randrange(1, radius)
            if d not in dist:
                dist[d] = True
                added += 1
    closest = sorted(dist)[:K]
    is_sybil = [dist[d] for d in closest]

    h = 0.5 if clustered else 0.05
    in_provider = sum(rng.random() < h for _ in range(K))
    order = list(range(K))
    rng.shuffle(order)
    if attacked:
        order.sort(key=lambda i: not is_sybil[i])  # stable: Sybils first
    provider_idx = set(order[:in_provider])

    taken: set = set()
    pa, pb = _slash16(rng, taken)
    peers = []
    for i, d in enumerate(closest):
        a, b = (pa, pb) if i in provider_idx else _slash16(rng, taken)
        peers.append(
            {
                "id": hx(target ^ d),
                "cpl": cpl(target, target ^ d),
                "ip": f"{a}.{b}.{rng.randint(0, 255)}.{rng.randint(1, 254)}",
                "sybil": is_sybil[i],
            }
        )
    rng.shuffle(peers)

    if show_cpl:
        header = "peer_id   cpl  ip"
        rows = "\n".join(f"{p['id']}  {p['cpl']:>3}  {p['ip']}" for p in peers)
        legend = "(cpl = number of leading bits the peer id shares with the key)\n"
    else:
        header = "peer_id   ip"
        rows = "\n".join(f"{p['id']}  {p['ip']}" for p in peers)
        legend = ""
    return {
        "cell": CELLS[(attacked, clustered)],
        "label": "ECLIPSED" if attacked else "HEALTHY",
        "target": hx(target),
        "n": n,
        "sybils_inserted": sybils if attacked else 0,
        "grind": grind if attacked else 0,
        "in_provider": in_provider,
        "peers": peers,
        "suspects": ",".join(sorted(p["id"] for p in peers if p["sybil"])),
        "prompt": VERDICT_PROMPT.format(
            target=hx(target),
            n=n,
            h_pct=round(h * 100),
            prefix=f"{pa}.{pb}",
            header=header,
            rows=rows,
            legend=legend,
        ),
    }


def verdict_items(n: int = 80, seed: int = 2026, show_cpl: bool = True) -> list[dict]:
    """Balanced 2x2: n/4 items per cell."""
    rng = random.Random(f"verdict-{seed}")
    cells = [(a, c) for a in (True, False) for c in (True, False)]
    out = []
    for i in range(n):
        attacked, clustered = cells[i % 4]
        item = gen_verdict_item(rng, attacked, clustered, show_cpl=show_cpl)
        item["item_id"] = f"verdict-{i:03d}"
        out.append(item)
    return out


def verdict_sweep(
    per_point: int = 10,
    seed: int = 2026,
    grinds=(0, 1, 2, 3, 4, 6),
    sybil_counts=(4, 8, 12, 16, 20),
) -> list[dict]:
    """Attacked items only, across attacker stealth. Pair with verdict_items
    for the honest side when computing detection rates."""
    rng = random.Random(f"sweep-{seed}")
    out = []
    for g in grinds:
        for s in sybil_counts:
            for j in range(per_point):
                item = gen_verdict_item(rng, True, bool(j % 2), sybils=s, grind=g)
                item["item_id"] = f"sweep-g{g}-s{s:02d}-{j:02d}"
                out.append(item)
    return out


# --------------------------------------------------------------------------
# Task 4: admission policy
# --------------------------------------------------------------------------

ADMISSION_PROMPT = """\
A Kademlia node enforces a subnet-diversity limit when admitting peers to its routing table.

Policy:
- Peers are grouped by subnet: IPv4 addresses by /24, IPv6 addresses by /48.
- A bucket may hold at most {b_limit} peers from the same group.
- The whole routing table may hold at most {t_limit} peers from the same group.
- A candidate is admitted only if neither limit would be exceeded after adding it. Every bucket has free slots.

Current routing table:
bucket  peer_id   ip
{rows}

Candidate: peer {cand_id}, ip {cand_ip}, belongs in bucket {cand_bucket}.

You may show your working. Finish with one final line in exactly this form:
DECISION: ADMIT or REJECT"""

ADMISSION_KINDS = (
    "v4_under",
    "v4_bucket_full",
    "v4_table_full",
    "v4_near_miss",  # same /16, different /24 -> admit
    "v6_under",
    "v6_bucket_full",  # same /48, different /64 -> reject
    "v6_table_full",
    "v6_near_miss",  # same /32, different /48 -> admit
)


def subnet_group(ip: str, v6_prefix: int = 48) -> str:
    addr = ipaddress.ip_address(ip)
    bits = 24 if addr.version == 4 else v6_prefix
    return str(ipaddress.ip_network(f"{ip}/{bits}", strict=False))


def admission_decision(table, cand_ip, cand_bucket, b_limit, t_limit, v6_prefix=48):
    """table: list of (bucket, ip). Returns 'ADMIT' or 'REJECT'."""
    g = subnet_group(cand_ip, v6_prefix)
    same = [b for b, ip in table if subnet_group(ip, v6_prefix) == g]
    in_bucket = sum(1 for b in same if b == cand_bucket)
    ok = in_bucket + 1 <= b_limit and len(same) + 1 <= t_limit
    return "ADMIT" if ok else "REJECT"


def _v4(rng, a, b, c=None):
    c = rng.randint(0, 255) if c is None else c
    return f"{a}.{b}.{c}.{rng.randint(1, 254)}"


def _v6(rng, h1, h2, h3=None):
    """Address in h1:h2:h3::/48 with a random /64 and host part."""
    h3 = rng.getrandbits(16) if h3 is None else h3
    value = (h1 << 112) | (h2 << 96) | (h3 << 80) | (rng.getrandbits(16) << 64)
    return str(ipaddress.IPv6Address(value | rng.randint(1, 0xFFFF)))


def gen_admission_item(rng: random.Random, kind: str) -> dict:
    v6 = kind.startswith("v6")
    shape = kind[3:]
    b_limit, t_limit = 2, rng.choice((3, 4))
    buckets = rng.sample(range(2, 14), 3)
    cand_bucket = buckets[0]

    taken: set = set()
    a, b = _slash16(rng, taken)
    c = rng.randint(0, 255)
    h1, h2, h3 = rng.randint(0x2001, 0x2A0F), rng.getrandbits(16), rng.getrandbits(16)

    used: set = set()

    def fresh(make):
        while True:
            ip = make()
            if ip not in used:
                used.add(ip)
                return ip

    def same_group():
        return fresh(lambda: _v6(rng, h1, h2, h3) if v6 else _v4(rng, a, b, c))

    def near_group():  # looks related, different group
        if v6:
            return fresh(lambda: _v6(rng, h1, h2, (h3 + rng.randint(1, 65535)) % 65536))
        return fresh(lambda: _v4(rng, a, b, (c + rng.randint(1, 255)) % 256))

    def unrelated():
        if rng.random() < 0.5:
            return _v4(rng, *_slash16(rng, taken))
        return _v6(rng, rng.randint(0x2001, 0x2A0F), rng.getrandbits(16))

    if shape == "bucket_full":
        n_bucket, n_other = b_limit, rng.randint(0, t_limit - b_limit - 1)
    elif shape == "table_full":
        n_bucket = rng.randint(0, b_limit - 1)
        n_other = t_limit - n_bucket
    else:  # under, near_miss
        n_bucket = rng.randint(0, b_limit - 1)
        n_other = rng.randint(0, t_limit - 1 - n_bucket)

    table = [(cand_bucket, same_group()) for _ in range(n_bucket)]
    table += [(rng.choice(buckets[1:]), same_group()) for _ in range(n_other)]
    if shape == "near_miss":  # enough look-alikes to trip a sloppy grouping
        table += [(cand_bucket, near_group()) for _ in range(b_limit)]
        table += [(rng.choice(buckets[1:]), near_group()) for _ in range(t_limit)]
    while len(table) < 12:
        table.append((rng.choice(buckets), unrelated()))
    rng.shuffle(table)
    table.sort(key=lambda row: row[0])

    cand_ip = same_group()
    rows = "\n".join(
        f"{bk:>6}  {hx(rng.getrandbits(BITS))}  {ip}" for bk, ip in table
    )
    args = (table, cand_ip, cand_bucket, b_limit, t_limit)
    return {
        "kind": kind,
        "answer": admission_decision(*args),
        # what grouping IPv6 by /64 (a common mistake) would decide
        "answer_if_v6_slash64": admission_decision(*args, v6_prefix=64),
        "b_limit": b_limit,
        "t_limit": t_limit,
        "prompt": ADMISSION_PROMPT.format(
            b_limit=b_limit,
            t_limit=t_limit,
            rows=rows,
            cand_id=hx(rng.getrandbits(BITS)),
            cand_ip=cand_ip,
            cand_bucket=cand_bucket,
        ),
    }


def admission_items(n: int = 64, seed: int = 2026) -> list[dict]:
    rng = random.Random(f"admission-{seed}")
    out = []
    for i in range(n):
        item = gen_admission_item(rng, ADMISSION_KINDS[i % len(ADMISSION_KINDS)])
        item["item_id"] = f"admission-{i:03d}"
        out.append(item)
    return out


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
