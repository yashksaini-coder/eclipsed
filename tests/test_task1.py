"""Run the built Kaggle file end to end against scripted models."""
import pathlib
import re

import pytest

import kaggle_benchmarks as kbench
from kaggle_benchmarks import ExecutionMode, actors, clients, config, contexts
from kaggle_benchmarks.llm_messages import LLMMessage

DIST = pathlib.Path(__file__).parent.parent / "dist" / "task1_xor_distance.py"


class Scripted(actors.LLMChat):
    def __init__(self, policy, name):
        super().__init__(name=name)
        self.policy = policy

    def invoke(self, messages, tools=None, **kwargs):
        text = str(messages[-1].content)
        target = int(re.search(r"Target key: (\w{8})", text).group(1), 16)
        peers = re.search(r"Peers:\n((?:\w{8}\n)+)", text).group(1).split()
        return LLMMessage(sender=self, content=self.policy(target, peers))


POLICIES = {
    "oracle": lambda t, ps: "ANSWER: " + min(ps, key=lambda p: int(p, 16) ^ t),
    "numeric": lambda t, ps: "Closest by subtraction.\nANSWER: 0x"
    + min(ps, key=lambda p: abs(int(p, 16) - t)).upper(),
    "rambler": lambda t, ps: "I think it is probably the first one.",
}
EXPECT = {  # accuracy, agree, trap, took_decoy, unparsed
    "oracle": (1.0, 1.0, 1.0, 0.0, 0.0),
    "numeric": (0.5, 1.0, 0.0, 1.0, 0.0),
    "rambler": (0.0, 0.0, 0.0, 0.0, 1.0),
}


def run_dist(llm, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    with contexts.enter():
        config.execution_mode = ExecutionMode.TESTING
        config.interactive_mode = False
        config.apply()
        monkeypatch.setattr(kbench, "client", clients.InMemoryClient())
        monkeypatch.setattr(kbench, "llm", llm)
        ns = {}
        exec(compile(DIST.read_text(), str(DIST), "exec"), ns)
    return ns


@pytest.mark.parametrize("who", POLICIES)
def test_task_scores_scripted_models(who, monkeypatch, tmp_path, capsys):
    ns = run_dist(Scripted(POLICIES[who], who), monkeypatch, tmp_path)
    acc, agree, trap, decoy, unparsed = EXPECT[who]
    score, ci = ns["run"].result
    assert score == pytest.approx(acc)
    assert ci == pytest.approx(1.96 * (acc * (1 - acc) / 60) ** 0.5)
    printed = eval(capsys.readouterr().out.strip().splitlines()[-1])
    assert printed["agree_accuracy"] == agree and printed["trap_accuracy"] == trap
    assert printed["took_numeric_decoy"] == decoy and printed["unparsed"] == unparsed
    assert printed["decoy_gap"] == pytest.approx(agree - trap) and printed["errored"] == 0


def test_items_that_keep_erroring_count_as_wrong(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr("time.sleep", lambda s: None)
    seen = set()

    def flaky(t, ps):  # every item fails on first contact, succeeds on retry
        if t not in seen:
            seen.add(t)
            raise TimeoutError("model proxy timed out")
        return POLICIES["oracle"](t, ps)

    always_down = lambda t, ps: (_ for _ in ()).throw(TimeoutError("down"))
    ns = run_dist(Scripted(always_down, "down"), monkeypatch, tmp_path)
    assert ns["run"].result[0] == 0.0
    assert eval(capsys.readouterr().out.strip().splitlines()[-1])["errored"] == 60

    ns = run_dist(Scripted(flaky, "flaky"), monkeypatch, tmp_path)
    printed = eval(capsys.readouterr().out.strip().splitlines()[-1])
    assert ns["run"].result[0] == 1.0 and printed["errored"] == 0  # retries recovered every item


def test_answer_parsing(monkeypatch, tmp_path):
    ns = run_dist(Scripted(POLICIES["oracle"], "oracle"), monkeypatch, tmp_path)
    parse = ns["parse_answer"]
    assert parse("ANSWER: 7f3a9c21") == "7f3a9c21"
    assert parse("**ANSWER:** `0x7F3A9C21`.") == "7f3a9c21"
    assert parse("ANSWER: 00000000 no wait\nANSWER: 7f3a9c21") == "7f3a9c21"
    assert parse("ANSWER: 7f3a9c2") is None and parse("the answer is 7f3a9c21") is None
