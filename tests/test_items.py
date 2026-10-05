"""Ground truth is re-derived here by a different route than the generator uses
(bit strings instead of integer XOR, a hand-rolled subnet match instead of
ipaddress), so a shared bug cannot hide."""
import collections
import ipaddress
import random
import re

from kadbench import baseline
import kadbench.gen as g


def bits(h):
    return bin(int(h, 16))[2:].zfill(32)


def xor_bits(a, b):
    return "".join("1" if x != y else "0" for x, y in zip(bits(a), bits(b)))


def prefix_len(a, b):
    n = 0
    for x, y in zip(bits(a), bits(b)):
        if x != y:
            break
        n += 1
    return n


def test_xor_items():
    items = g.xor_items(400, seed=7)
    assert collections.Counter(i["kind"] for i in items) == {"agree": 200, "trap": 200}
    for it in items:
        peers = it["peers"].split(",")
        assert len(set(peers)) == 8 and it["target"] not in peers
        # equal-length bit strings sort the same way as the integers they encode
        by_xor = min(peers, key=lambda p: xor_bits(p, it["target"]))
        by_num = min(peers, key=lambda p: abs(int(p, 16) - int(it["target"], 16)))
        assert it["answer"] == by_xor and it["decoy"] == by_num
        assert (by_xor != by_num) == (it["kind"] == "trap")
        assert all(p in it["prompt"] for p in peers) and it["target"] in it["prompt"]


def test_bucket_items():
    items = g.bucket_items(400, seed=7)
    for it in items:
        assert it["answer"] == prefix_len(it["local"], it["peer"])
        if it["kind"] == "boundary":
            assert abs(int(it["local"], 16) - int(it["peer"], 16)) <= 200
            assert it["answer"] <= 23
    assert len({i["answer"] for i in items if i["kind"] == "uniform"}) == 25


def test_verdict_items_structure():
    items = g.verdict_items(400, seed=7)
    assert set(collections.Counter(i["cell"] for i in items).values()) == {100}
    for it in items:
        peers = it["peers"]
        assert len(peers) == 20 and len({p["id"] for p in peers}) == 20
        for p in peers:
            assert p["cpl"] == prefix_len(p["id"], it["target"])
            assert p["id"] in it["prompt"] and p["ip"] in it["prompt"]
        n_sybil = sum(p["sybil"] for p in peers)
        assert (n_sybil > 0) == (it["label"] == "ECLIPSED")
        assert it["suspects"].count(",") + 1 == n_sybil or (n_sybil == 0 and it["suspects"] == "")
        prefix = re.search(r"inside (\d+\.\d+)\.0\.0/16", it["prompt"]).group(1)
        assert sum(p["ip"].startswith(prefix + ".") for p in peers) == it["in_provider"]
        others = [p["ip"].rsplit(".", 2)[0] for p in peers if not p["ip"].startswith(prefix + ".")]
        assert len(others) == len(set(others))  # every non-provider peer in its own /16


def test_honest_sets_match_brute_force():
    """The order-statistics shortcut must agree with literally sampling n IDs."""
    rng = random.Random(1)
    n, trials = 1000, 3000
    fast = collections.Counter()
    slow = collections.Counter()
    for _ in range(trials):
        fast.update(32 - d.bit_length() for d in g.honest_closest(rng, n))
        ds = sorted(rng.getrandbits(32) for _ in range(n))[:20]
        slow.update(32 - d.bit_length() for d in ds)
    expected = baseline.expected_cpl_pmf(n)
    for c in range(4, 12):
        f, s = fast[c] / (20 * trials), slow[c] / (20 * trials)
        assert abs(f - s) < 0.01, (c, f, s)
        assert abs(expected[c] - s) < 0.01, (c, expected[c], s)
    assert abs(sum(expected) - 1) < 1e-6


def test_ip_column_carries_no_label():
    """No rule that reads only the IP column should beat a coin flip."""
    items = g.verdict_items(4000, seed=11)
    feats = lambda it: (
        it["in_provider"],
        max(collections.Counter(p["ip"].split(".")[0] for p in it["peers"]).values()),
    )
    # best possible lookup-table classifier on the IP-only features, fit on half, scored on the other half
    train, test = items[:2000], items[2000:]
    votes = collections.defaultdict(collections.Counter)
    for it in train:
        votes[feats(it)][it["label"]] += 1
    guess = lambda it: (votes[feats(it)].most_common(1) or [("HEALTHY", 0)])[0][0]
    acc = sum(guess(it) == it["label"] for it in test) / len(test)
    assert 0.46 < acc < 0.54, acc


def test_reference_detectors():
    items = g.verdict_items(400, seed=7)
    kl = baseline.score(items, baseline.kl_detector)
    ip = baseline.score(items, baseline.ip_heuristic)
    assert min(kl[c] for c in baseline.AGREE + baseline.CONFLICT) >= 0.95
    assert abs(kl["decoy_gap"]) < 0.05
    assert ip["decoy_gap"] > 0.9


def manual_group(ip):
    if ":" in ip:
        full = ipaddress.ip_address(ip).exploded.split(":")
        return ("v6", tuple(full[:3]))
    return ("v4", tuple(ip.split(".")[:3]))


def test_admission_items():
    items = g.admission_items(800, seed=7)
    by_kind = collections.defaultdict(set)
    slash64_flips = 0
    for it in items:
        rows = re.findall(r"^\s*(\d+)\s+[0-9a-f]{8}\s+(\S+)$", it["prompt"], re.M)
        assert len(rows) >= 12 and len({ip for _, ip in rows}) == len(rows)
        cand_ip, cand_bucket = re.search(r"ip (\S+), belongs in bucket (\d+)", it["prompt"]).groups()
        assert cand_ip not in {ip for _, ip in rows}
        same = [b for b, ip in rows if manual_group(ip) == manual_group(cand_ip)]
        ok = same.count(cand_bucket) + 1 <= it["b_limit"] and len(same) + 1 <= it["t_limit"]
        assert it["answer"] == ("ADMIT" if ok else "REJECT")
        by_kind[it["kind"]].add(it["answer"])
        slash64_flips += it["answer"] != it["answer_if_v6_slash64"]
    expect = {k: {"REJECT"} if "full" in k else {"ADMIT"} for k in g.ADMISSION_KINDS}
    assert dict(by_kind) == expect
    assert slash64_flips >= 190  # nearly every v6 reject item flips under /64 grouping


def test_seeds_are_deterministic():
    for fn in g.FAMILIES.values():
        assert fn(seed=3) == fn(seed=3)
        assert fn(seed=3) != fn(seed=4)
