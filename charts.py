#!/usr/bin/env python3
"""Build the article's inline SVG charts from results/*.csv.

  python3 charts.py      # writes results/charts/*.html
"""
import csv, os
from heldout_charts import bars, dot_ci, dumbbell

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "results", "charts")
NAMES = {"openai/gpt-6.1-sol": "GPT-6.1 Sol", "anthropic/claude-sonnet-5.5": "Claude Sonnet 5.5",
         "google/gemini-4-argon": "Gemini 4 Argon", "x-ai/grok-4.7": "Grok 4.7", "moonshotai/kimi-k3.1-pro": "Kimi K3.1 Pro",
         "deepseek/deepseek-v4.1-flash": "DeepSeek v4.1 Flash", "mistralai/mistral-large-4-0": "Mistral Large 4",
         "z-ai/glm-5.3": "GLM 5.3", "qwen/qwen3.8-max-0902": "Qwen 3.8 Max", "amazon/nova-2-lite-v1": "Amazon Nova 2 Lite"}
CATS = {"llm_apis": "LLM APIs", "ai_assistants": "AI assistants", "coding_tools": "Coding tools", "search": "Search engines",
        "cloud": "Cloud hosting", "browsers": "Web browsers", "productivity": "Office suites"}
read = lambda f: list(csv.DictReader(open(os.path.join(ROOT, "results", f))))
S = read("self-preference-2026-10-summary.csv")
C = read("self-preference-2026-10-categories.csv")

# Figure 1: overall gap per model
rows = []
for r in sorted(S, key=lambda r: -float(r["first_gap_pts"])):
    v, lo, hi = float(r["first_gap_pts"]), float(r["first_gap_ci_low"]), float(r["first_gap_ci_high"])
    rows.append({"label": NAMES[r["model"]], "sub": f'{float(r["own_first_pct"]):.0f}% vs {float(r["others_first_pct"]):.0f}% for other models',
                 "value": v, "low": lo, "high": hi, "strong": lo > 0 or hi < 0, "text": f"{v:+.1f} pts" if v else "0.0 pts",
                 "tip": f'{NAMES[r["model"]]}: {v:+.1f} points (95% CI {lo:+.1f} to {hi:+.1f})'})
fig1 = dot_ci(rows, -15, 25, 5, lambda t: f"{t:+.0f}" if t else "0", "Self-preference gap by model", rowh=44)

# Figure 2: the category cases with the largest gaps (own model vs the other nine)
pick = [("anthropic/claude-sonnet-5.5", "llm_apis"), ("anthropic/claude-sonnet-5.5", "coding_tools"),
        ("moonshotai/kimi-k3.1-pro", "ai_assistants"), ("moonshotai/kimi-k3.1-pro", "llm_apis"),
        ("openai/gpt-6.1-sol", "llm_apis"), ("amazon/nova-2-lite-v1", "cloud"), ("google/gemini-4-argon", "search")]
idx = {(r["model"], r["category"]): r for r in C}
rows = []
for m, c in pick:
    r = idx[(m, c)]
    own, oth = float(r["own_first_pct"]), float(r["others_first_pct"])
    rows.append({"label": f"{NAMES[m]}", "sub": CATS[c], "a": oth, "b": own, "a_text": f"{oth:.0f}%", "b_text": f"{own:.0f}%",
                 "tip_a": f"Other 9 models, {CATS[c]}: {oth:.0f}% list {r['maker']} first", "tip_b": f"{NAMES[m]}, {CATS[c]}: {own:.0f}% list {r['maker']} first"})
fig2 = dumbbell(rows, 0, 100, 20, lambda t: f"{t:.0f}%", "Share of answers listing the maker first, own model vs other models")

# Figure 3: disclosure among answers that recommend the model's own maker (models with 40+ such answers)
rows = []
for r in sorted([r for r in S if int(r["self_recommending_answers"]) >= 40], key=lambda r: -float(r["discloses_own_maker_pct"])):
    v, n = float(r["discloses_own_maker_pct"]), int(r["self_recommending_answers"])
    rows.append({"label": NAMES[r["model"]], "value": v, "text": f"{v:.0f}%", "strong": v >= 50,
                 "tip": f"{NAMES[r['model']]}: disclosed in {round(v * n / 100)} of {n} self-recommending answers"})
fig3 = bars(rows, 100, 25, lambda t: f"{t:.0f}%", "Share of self-recommending answers that disclose the connection")

os.makedirs(OUT, exist_ok=True)
for name, svg in (("fig1-gap", fig1), ("fig2-categories", fig2), ("fig3-disclosure", fig3)):
    open(os.path.join(OUT, name + ".html"), "w").write(svg)
print("wrote", OUT)
