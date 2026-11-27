"""
judge_agreement.py - measures how often the automatic judge agrees with a human.

A person labels a sample of saved answers as acceptable or not; we compare with
the judge's pass/fail. This answers the evaluator's hardest question - "why
should I trust your score?" - with a number and the list of disagreements:
  false pass = judge said pass, human said wrong   (judge too lenient)
  false fail = judge said fail, human said fine     (judge too strict)
"""

import json
import random
from pathlib import Path

from pydantic import BaseModel

from forgeai import config
from forgeai.roles.judge import CaseResult

LABELS_FILE = "human_labels.json"


class HumanLabel(BaseModel):
    scorecard: str
    id: str
    question: str
    answer: str
    judge: str          # "pass" | "fail"
    human_ok: bool


class Agreement(BaseModel):
    labelled: int
    agree: int
    rate: float
    false_pass: list[str]
    false_fail: list[str]


def choose_sample(results: list[CaseResult], n: int) -> list[CaseResult]:
    """All judge failures (most likely to be wrong) plus passes, up to n, in a fixed order."""
    scored = [r for r in results if r.status in ("pass", "fail")]
    fails = [r for r in scored if r.status == "fail"]
    passes = [r for r in scored if r.status == "pass"]
    random.Random(config.SEED).shuffle(passes)
    sample = (fails + passes)[:n]
    return sorted(sample, key=lambda r: r.id)


def agreement(labels: list[HumanLabel]) -> Agreement:
    agree = [l for l in labels if (l.judge == "pass") == l.human_ok]
    return Agreement(
        labelled=len(labels), agree=len(agree),
        rate=round(len(agree) / len(labels), 3) if labels else 0.0,
        false_pass=[l.id for l in labels if l.judge == "pass" and not l.human_ok],
        false_fail=[l.id for l in labels if l.judge == "fail" and l.human_ok],
    )


def load_labels(run_dir: str | Path) -> list[HumanLabel]:
    path = Path(run_dir) / LABELS_FILE
    if not path.is_file():
        return []
    return [HumanLabel.model_validate(x) for x in json.loads(path.read_text(encoding="utf-8"))]


def save_labels(labels: list[HumanLabel], run_dir: str | Path) -> Path:
    path = Path(run_dir) / LABELS_FILE
    path.write_text(json.dumps([l.model_dump() for l in labels], indent=2, ensure_ascii=False), encoding="utf-8")
    return path
