# Do AI models recommend their own company's products?

Code and data for the [Heldout Labs](https://heldoutlabs.com) study **[Do AI models recommend their own company's products? We asked 10 models 1,680 times](https://heldoutlabs.com/ai-models-recommend-own-products)** (October 2026).

We asked 10 models 56 brand-neutral buying-advice questions in 7 product categories, 3 times each, and measured how often each model lists its own maker's product first compared with the other nine models.

## Results

| Model | Own maker listed first | Other models | Gap, points (95% CI) |
|---|---|---|---|
| Kimi K3.1 Pro | 15.5% | 0.0% | +15.5 (7.7 to 23.8) |
| Claude Sonnet 5.5 | 22.0% | 8.2% | +13.8 (6.5 to 22.2) |
| Amazon Nova 2 Lite | 14.9% | 8.3% | +6.6 (2.8 to 11.2) |
| GPT-6.1 Sol | 18.5% | 12.7% | +5.8 (1.4 to 10.3) |
| Qwen 3.8 Max | 4.2% | 0.1% | +4.1 (-0.1 to 9.5) |
| Grok 4.7 | 1.8% | 0.0% | +1.8 (0.0 to 5.4) |
| GLM 5.3 | 0.0% | 0.0% | 0.0 |
| Mistral Large 4 | 0.0% | 0.1% | -0.1 (-0.4 to 0.0) |
| DeepSeek v4.1 Flash | 0.0% | 0.3% | -0.3 (-0.8 to 0.0) |
| Gemini 4 Argon | 18.5% | 23.2% | -4.8 (-12.0 to 2.4) |

Per-category results and disclosure rates are in [`results/`](results/) and in the article.

## Repository

| Path | Contents |
|---|---|
| `questions.json` | The 56 questions, by category |
| `study.py` | Runs the models, extracts recommendations, and computes the analysis |
| `export.py` | Writes the published CSVs, with bootstrap confidence intervals |
| `data/answers.jsonl` | All 1,680 raw answers, with token usage and cost |
| `data/extractions.jsonl` | Every extractor output (3 extractor models) |
| `data/consensus.jsonl` | Per-answer majority vote: first-listed vendor, recommended vendors, disclosure |
| `results/*.csv` | The published summary, per-category and per-answer tables |

## Reproduce

The analysis needs only Python 3 (standard library) and the data in this repo:

```sh
python3 study.py analyze          # data/consensus.jsonl and data/summary.json from the raw data
python3 export.py results         # results/*.csv, including confidence intervals
```

Re-running the models needs an [OpenRouter](https://openrouter.ai) key in `.env` (see `.env.example`). Model responses are not deterministic, so new runs will differ in detail.

```sh
python3 study.py run --models openai/gpt-6.1-sol,anthropic/claude-sonnet-5.5 \
  --categories ai_assistants,llm_apis --samples 3 --budget 5 --max-tokens 16000
python3 study.py drop-empty                       # re-ask answers that came back empty
python3 study.py extract --budget 2               # two extractor models per answer
python3 study.py extract --tiebreak --budget 1    # third extractor where they disagree
python3 study.py analyze
```

`--budget` is a hard spending cap in USD, enforced from OpenRouter's per-request cost.

## Method in brief

- **Models:** one current model per lab, via OpenRouter, default settings, no system prompt, answers collected on 2026-10-28.
- **Extraction:** Claude Haiku 5.5 and DeepSeek v4.1 Flash list each answer's recommendations in order; GPT-6 Luna breaks ties on 200 answers where they disagree or fail. Products map to companies through the keyword list in `study.py`.
- **Measure:** share of a model's answers whose first-listed product belongs to its maker, minus that share for the other nine models, averaged over questions. 95% intervals resample questions, 5,000 draws.

Full methodology and limitations: [heldoutlabs.com/ai-models-recommend-own-products](https://heldoutlabs.com/ai-models-recommend-own-products).

## License

Code: [MIT](LICENSE). Data (`data/`, `results/`): [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Please cite:

> symbatix (2026). "Do AI models recommend their own company's products? We asked 10 models 1,680 times." Heldout Labs. https://heldoutlabs.com/ai-models-recommend-own-products
