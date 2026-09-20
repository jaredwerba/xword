# xword

Crossword agent for the Nebius FDE take-home.

**Python owns the grid. Wordlist + Jev rank tight patterns. Token Factory `fill_region` writes leftovers. Tavily is last resort.**

Live: [www.jwerba.com/xword](https://www.jwerba.com/xword)  
Source: this repo. The earlier LangGraph agent stays at [nebius-xword.vercel.app](https://nebius-xword.vercel.app) as the generation-loop baseline.

## Why this loop

The first agent asked DeepSeek V4 Pro to call `get_state` / `fill_slot` / `submit` for every judgment. A 13×13 took **98 turns, 996s, 2.44M tokens**. Most of those tokens were not answers — they were routing.

This agent never spends a Token Factory turn on “which slot” or “are we done.”

| Layer | Product | Job |
|---|---|---|
| Engine | Python `Grid.fill_slot` | Crossings. A wrong word cannot corrupt state. |
| Constrained | Wordlist + Jev (`typesafe-ai/jev`) | Unique / Choice among ≤12 after ≥2 letters. Crossing clues in Jev state. |
| Leftovers | [Nebius Token Factory](https://docs.tokenfactory.nebius.com) `fill_region` | One generation call per leftover component. Not filtered to 3/4/5/7. |
| Last resort | [Tavily](https://docs.tavily.com) | Only slots still empty after region fill. Single hit skips Jev. |
| Observability | LangSmith | One trace per solve — Blueprint [recipe 07](https://github.com/nebius/nebius-partner-cookbook/tree/main/cookbooks/07-observability-langsmith) |

Jev is a ranker for a short list once the pattern is tight. Token Factory writes the leftover grid. Tavily is not the default writer.

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

Live race vs [nebius-xword](https://nebius-xword.vercel.app) (same puzzle ids, DeepSeek V4 Pro LangGraph vs search-jev):

| Puzzle | LangGraph wall / TF tokens | search-jev wall / TF tokens | Winner |
|---|---|---|---|
| 3×3 | 12.1s / 3,116 | 6.0–6.8s / **0** | search-jev |
| mini 5×5 | 15.2s / 12,764 | 5.5–10s / **0** | search-jev |
| 5×5-b | ~19s / 14.7k (recorded) | **7.9s / 0** | search-jev |
| 16×16 (96 slots) | **190s / 242,739 / 100%** | **166s / 1,527 / 92% letters** | search-jev faster + 159× fewer TF tokens; LangGraph matches the key |
| 32×32 (384 slots) | **786s / 2,126,397 / 100%** | **211s / 3,277 / 96.5% letters** | search-jev **3.7×** wall, **649×** fewer TF tokens |
| 64×64 (1536 slots) | **2009s / 17.7M TF / 98.8% letters** | **511s / 3,598 TF / 97.4% letters, not submitted** | search-jev **3.9×** wall, **4,908×** fewer TF tokens; USD **$3.85 vs $30.90** |

Full USD (Token Factory $1.75/MTok + Jev $0.042/MTok + Tavily $0.008/search), 64×64, and the tiled-3×3 caveat: **[eval/SCALE.md](eval/SCALE.md)**. 16/32/64 boards are independent 3×3s, not interlocking Sundays. `solved` is key-match; a valid synonym fill is a miss.

Token Factory is the cost axis of the old demo. search-jev is **$0 TF** on the interlocking minis. At 32×32 Tavily ($1.69) dominates search-jev USD and it is still cheaper than LangGraph ($1.71 vs $3.72). At 64×64 that gap is **$3.85 vs $30.90**.

LangSmith project `xword`. Traced mini 5×5: [public run](https://smith.langchain.com/public/4c22b686-1477-4d6a-9df7-9d5339d4ff0c/r). Scale 32/64 tracing is off (monthly unique-traces cap).

## 60-second demo script

1. Engine owns the grid; `fill_slot` rejects a crossing conflict.
2. Wordlist + Jev rank a slot only after ≥2 letters (tight pattern).
3. Token Factory `fill_region` writes the leftover component in one generation call.
4. Tavily runs only if that region pass left empties.

## License

MIT
