#!/usr/bin/env python3
"""Export publishable CSVs from data/consensus.jsonl and data/answers.jsonl.

  python3 export.py OUT_DIR

Writes, with prefix self-preference-2026-10:
  -summary.csv     one row per model: self-preference gaps with 95% bootstrap CIs, disclosure rate
  -categories.csv  one row per model x category: first-listed rates and gap with CI
  -answers.csv     one row per answer: question, consensus first-listed vendor, all recommended vendors
  -transcripts.jsonl.gz  every raw answer text
CIs resample the 8 questions of a category (or all 56 questions) with replacement, 5000 draws.
"""
import csv, gzip, json, os, random, sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
PREFIX = "self-preference-2026-10"
DRAWS = 5000

rows = [json.loads(l) for l in open(os.path.join(HERE, "data/consensus.jsonl"))]
answers = {(a["model"], a["qid"], a["sample"]): a for a in map(json.loads, open(os.path.join(HERE, "data/answers.jsonl")))}
models = sorted({r["model"] for r in rows})
maker = {r["model"]: r["maker"] for r in rows}
cats = sorted({r["category"] for r in rows})


def gap(model, field, category=None, seed=7):
    """Mean over questions of (own-model hit rate - other-models hit rate), with a question-level bootstrap CI."""
    v, per_q = maker[model], defaultdict(lambda: ([], []))
    for r in rows:
        if category and r["category"] != category:
            continue
        hit = r["first"] == v if field == "first" else v in r["mentioned"]
        per_q[r["qid"]][0 if r["model"] == model else 1].append(hit)
    pairs = [(sum(a) / len(a), sum(b) / len(b)) for a, b in per_q.values()]
    mean = lambda s: sum(x - y for x, y in s) / len(s)
    own = sum(x for x, _ in pairs) / len(pairs)
    others = sum(y for _, y in pairs) / len(pairs)
    rng = random.Random(seed)
    boots = sorted(mean([rng.choice(pairs) for _ in pairs]) for _ in range(DRAWS))
    return own, others, mean(pairs), boots[int(0.025 * DRAWS)], boots[int(0.975 * DRAWS) - 1]


def pct(x):
    return round(100 * x, 1)


out = sys.argv[1]
os.makedirs(out, exist_ok=True)

with open(os.path.join(out, f"{PREFIX}-summary.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["model", "maker", "own_first_pct", "others_first_pct", "first_gap_pts", "first_gap_ci_low", "first_gap_ci_high",
                "own_mentioned_pct", "others_mentioned_pct", "mentioned_gap_pts", "mentioned_gap_ci_low", "mentioned_gap_ci_high",
                "clear_single_pick_pct", "self_recommending_answers", "discloses_own_maker_pct"])
    for m in models:
        mine = [r for r in rows if r["model"] == m]
        promo = [r for r in mine if maker[m] in r["mentioned"]]
        o1, b1, g1, l1, h1 = gap(m, "first")
        o2, b2, g2, l2, h2 = gap(m, "mentioned")
        w.writerow([m, maker[m], pct(o1), pct(b1), pct(g1), pct(l1), pct(h1), pct(o2), pct(b2), pct(g2), pct(l2), pct(h2),
                    pct(sum(r["clear_pick"] for r in mine) / len(mine)), len(promo),
                    pct(sum(r["disclosed"] for r in promo) / len(promo)) if promo else ""])

with open(os.path.join(out, f"{PREFIX}-categories.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["model", "maker", "category", "own_first_pct", "others_first_pct", "first_gap_pts", "ci_low", "ci_high"])
    for m in models:
        for c in cats:
            o, b, g, lo, hi = gap(m, "first", c)
            w.writerow([m, maker[m], c, pct(o), pct(b), pct(g), pct(lo), pct(hi)])

with open(os.path.join(out, f"{PREFIX}-answers.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["model", "maker", "category", "question_id", "question", "sample", "first_listed_vendor",
                "recommended_vendors", "clear_single_pick", "discloses_own_maker"])
    for r in rows:
        a = answers[(r["model"], r["qid"], r["sample"])]
        w.writerow([r["model"], r["maker"], r["category"], r["qid"], a["question"], r["sample"], r["first"] or "",
                    ";".join(r["mentioned"]), r["clear_pick"], r["disclosed"]])

with gzip.open(os.path.join(out, f"{PREFIX}-transcripts.jsonl.gz"), "wt") as f:
    for a in answers.values():
        f.write(json.dumps({k: a.get(k) for k in ("model", "category", "qid", "question", "sample", "provider", "text")}) + "\n")

print("wrote", sorted(os.listdir(out)))
