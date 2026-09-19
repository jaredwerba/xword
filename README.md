# xword

Crossword agent for the Nebius FDE take-home.

**Tavily grounds. Jev ranks. Token Factory writes leftovers. Python owns the grid.**

Live: [www.jwerba.com/xword](https://www.jwerba.com/xword)  
Source: this repo. The earlier LangGraph agent stays at [nebius-xword.vercel.app](https://nebius-xword.vercel.app) as the generation-loop baseline.

## Why this loop

The first agent asked DeepSeek V4 Pro to call `get_state` / `fill_slot` / `submit` for every judgment. A 13×13 took **98 turns, 996s, 2.44M tokens**. Most of those tokens were not answers — they were routing.

This agent never spends a Token Factory turn on “which slot” or “are we done.”

| Layer | Product | Job |
|---|---|---|
| Engine | Python `Grid.fill_slot` | Crossings. A wrong word cannot corrupt state. |
| Grounding | [Tavily](https://docs.tavily.com) | `crossword clue "…" N letters` — Blueprint [recipe 03](https://github.com/nebius/nebius-partner-cookbook/tree/main/cookbooks/03-real-time-data-tavily) |
| Decision | Jev (`typesafe-ai/jev`) | Choice among ≤12 candidates + `none` |
| Inference | [Nebius Token Factory](https://docs.tokenfactory.nebius.com) | Wordplay leftovers only |
| Observability | LangSmith | One trace per solve — Blueprint [recipe 07](https://github.com/nebius/nebius-partner-cookbook/tree/main/cookbooks/07-observability-langsmith) |

Jev is not a crossword solver. It does not invent first letters. Ranking a Tavily/wordlist shortlist is the job.

## Run

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env   # NEBIUS_API_KEY, TAVILY_API_KEY, LANGSMITH_API_KEY
# Jev: VERCEL_OIDC_TOKEN (vercel env pull) or AI_GATEWAY_API_KEY or TYPESAFE_API_KEY

python -m eval.run_eval --solver oracle          # no keys
python -m eval.run_eval --solver search-jev      # live agent
uvicorn api.index:app --reload                   # http://127.0.0.1:8000
```

Offline tests (no network):

```bash
pytest -q
```

## Evaluation methodology

Four solvers share one scorer (`eval/metrics.py`):

| Solver | What it proves |
|---|---|
| `empty` | Floor. Must be 0% letters. If not, the scorer is broken. |
| `oracle` | Ceiling. Must be 100%. |
| `backtrack` | Structure-only wordlist fill. Ignores clues (~9% letters on these fixtures). |
| `search-jev` | This agent. Report `solved`, `letter_accuracy`, `wall_ms`, `tf_tokens`, `tavily_calls`, `jev_calls`. |

Fixtures: `data/puzzles/example_3x3.json`, two 5×5s, `example_7x7.json`.

Gold bar (2026-09-19, live): **mini 5×5 `solved=true` in 7.7s**, **0 Token Factory tokens**, 2 Tavily calls, 8 Jev calls. 3×3: 6.9s, 0 TF tokens. LangGraph V4 Pro on the same 5×5 family was ~19–39s of generation.

## 60-second demo script

1. Engine owns the grid; `fill_slot` rejects a crossing conflict.
2. Tavily looks up the clue instead of asking a chat model to hallucinate CAT.
3. Jev ranks the shortlist in a few hundred milliseconds.
4. Token Factory only speaks when search and the wordlist are empty.

## License

MIT
