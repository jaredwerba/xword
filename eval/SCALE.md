# Scale races — wall time and token cost

Measured 2026-09-19 on this machine. Same fixtures, same Token Factory model (`deepseek-ai/DeepSeek-V4-Pro` at `https://api.tokenfactory.nebius.com/v1`).

These numbers are **performance and cost**, not crossword quality. The 16/32/64 boards are generated tiles, not published puzzles.

## What was raced

| Puzzle | Shape | Slots | Components | Kind |
|---|---|---|---|---|
| `example_3x3` | 3×3 | 6 | 1 interlocking | constructed mini |
| `example_mini_5x5` | 5×5 | 10 | 1 interlocking | constructed mini |
| `example_5x5_b` | 5×5 | 10 | 1 interlocking | constructed mini |
| `race_16x16` | 16×16 | 96 | 16 isolated 3×3s | `scripts/make_scale_puzzle.py` |
| `race_32x32` | 32×32 | 384 | 64 isolated 3×3s | same |
| `race_64x64` | 64×64 | 1536 | 256 isolated 3×3s | same |

The wordlist is 1663 words, lengths 3/4/5/7 only. A fully interlocking 16×16 with 6/8/16-letter slots will not fill from it. Scale fixtures stamp independent filled 3×3s separated by streets of blocks. Duplicate answers across tiles are allowed. Clues are short dictionary glosses.

`solved` in the benches is **letter match against the generator’s key**. A complete, crossing-legal 3×3 that answers the clues with a synonym (WAH vs SOB, PIG vs TOE) scores as a miss. LangGraph often matches the key because V4 Pro fills a whole mini in one generation turn. search-jev commits Tavily hits slot-by-slot.

## Agents

| Agent | Loop |
|---|---|
| **search-jev** (this repo) | Code picks the slot. Tavily grounds the clue. Jev ranks ≤12 candidates. Token Factory only writes leftovers / leftover-region repair. Python `fill_slot` owns crossings. ≥4 components run in a 4-worker pool; Tavily is capped at 2 in-flight requests. |
| **langgraph-tf** (`nebiusFDE`) | LangGraph + DeepSeek V4 Pro. The model calls `get_state` / `fill_slot` / `submit`. `history_window=48`. |

## Cost model (do not treat as a quote)

| Meter | Rate used here | Source of the rate |
|---|---|---|
| Token Factory tokens | **$1.75 / MTok** | take-home working figure for V4 Pro leftovers; not an official SKU table |
| Jev input tokens | **$0.042 / MTok** | TypeSafe/Gateway evaluation-model input |
| Tavily basic search | **$0.008 / search** | Tavily basic `search_depth` |

USD in `eval/bench-*.json` is `scripts/race_scale.py` applying those three rates. It is not a bill.

Scale 32/64 ran with `LANGCHAIN_TRACING_V2=false`. The 32×32 search-jev run had already blown LangSmith’s monthly unique-traces cap (`429` ingest). Demo traces stay on for 3×3/5×5.

## Interlocking minis (demo size)

Live race vs [nebius-xword.vercel.app](https://nebius-xword.vercel.app). search-jev TF tokens are **0**.

| Puzzle | LangGraph wall | LangGraph TF tok | LangGraph USD | search-jev wall | search-jev TF tok | Tavily | Jev in | search-jev USD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 3×3 | 12.1s | 3,116 | $0.0055 | 6.0–6.9s | **0** | 3 | ~3.1k | $0.024 |
| mini 5×5 | 15.2s | 12,764 | $0.022 | 5.5–7.7s | **0** | 2 | ~4.8k | $0.016 |
| 5×5-b | ~19s | ~14.7k | ~$0.026 | **7.9s** | **0** | n/a | n/a | Tavily-bound |

On interlocking minis, Tavily (a few basic searches) is the search-jev cost axis. Token Factory is the LangGraph cost axis. search-jev is slower-USD on the 3×3 only because three $0.008 searches beat 3k generation tokens; it is faster on the wall and cheaper on the 5×5.

## Generated tiles (the scale race)

Raw benches: `eval/bench-16x16.json`, `eval/bench-32x32.json`, `eval/bench-64x64.json`.

### Wall

| Size | slots | search-jev | langgraph-tf | wall speedup |
|---|---:|---:|---:|---:|
| 16×16 | 96 | **166s** | 190s | 1.1× |
| 32×32 | 384 | **211s** | 786s | **3.7×** |
| 64×64 | 1536 | **511s** (not submitted) | **2009s** (submitted) | **3.9×** |

32×32 search-jev is not 4× the 16×16 wall because components run 4-wide and Tavily is cached per `(clue, pattern)`. LangGraph is roughly linear in the number of minis (one generation turn per tile, plus retries).

### Token Factory (the old demo’s cost axis)

| Size | search-jev TF tok | search-jev TF $ | langgraph-tf TF tok | langgraph-tf TF $ | TF token ratio |
|---|---:|---:|---:|---:|---:|
| 16×16 | 1,527 | $0.0027 | 242,739 | $0.425 | **159×** |
| 32×32 | 3,277 | $0.0057 | 2,126,397 | $3.72 | **649×** |
| 64×64 | 3,598 | $0.0063 | 17,659,112 | $30.90 | **4,908×** |

search-jev Token Factory calls are leftovers plus `fill_region` repair, capped at 32 per-slot guesses (repair can add a few more). LangGraph spends a generation model on routing and on writing each mini.

### Full-stack USD (TF + Jev + Tavily)

| Size | search-jev Tavily | search-jev Jev in | search-jev USD | langgraph-tf USD | cheaper |
|---|---:|---:|---:|---:|---|
| 16×16 | 62 × $0.008 = $0.50 | 65,397 → $0.0027 | **$0.50** | $0.42 | LangGraph (~$0.08) |
| 32×32 | 211 × $0.008 = $1.69 | 280,894 → $0.012 | **$1.71** | $3.72 | **search-jev 2.2×** |
| 64×64 | 476 × $0.008 = $3.81 | 948,147 → $0.040 | **$3.85** | $30.90 | **search-jev 8.0×** |

Tavily basic search dominates search-jev USD as soon as the board has tens of unique clues. Jev is noise ($0.01 on 32×32). Token Factory is noise on search-jev and **the whole bill** on LangGraph.

If the comparison is “Token Factory tokens for the take-home,” search-jev wins at every size. If the comparison is “dollars to third parties at these working rates,” Tavily makes 16×16 a wash; 32×32 and 64×64 are still cheaper for search-jev because LangGraph’s generation bill grows with every mini.

## 32×32 detail

search-jev (`eval/bench-32x32.json`):

- 210,798 ms, 384 steps, submitted
- 211 Tavily, 411 Jev, 35 TF calls, 3,277 TF tokens, 280,894 Jev input tokens
- 96.5% letters / 93.2% words vs key (`solved: false`)
- $0.0057 TF + $0.012 Jev + $1.688 Tavily = **$1.71**

langgraph-tf:

- 785,556 ms, 82 turns, submitted
- 2,126,397 TF tokens
- 100% letters/words (`solved: true`)
- **$3.72** TF

During 32×32 search-jev, Tavily returned HTTP 429 (`excessive requests / production API keys`) until the solver retried with 1.5×2^attempt backoff and a process-wide Semaphore(2). LangSmith ingest 429’d on the unique-traces monthly cap; later scale runs disable tracing.

## 64×64

256 isolated 3×3s, 1,536 three-letter slots. Fixture `data/puzzles/race_64x64.json` (seed 16, same stamp as 16/32).

search-jev credited rerun (`eval/bench-64x64.json`):

- 511,465 ms, 1,536 steps, **not submitted** (leftover empties after the global 32 per-slot TF cap + `fill_region` repair)
- 476 Tavily, 1,343 Jev, 33 TF calls, 3,598 TF tokens, 948,147 Jev input tokens
- 97.4% letters / 95.8% words vs key (`solved: false`)
- $0.0063 TF + $0.040 Jev + $3.808 Tavily = **$3.85**

A prior attempt while Tavily pay-as-you-go was exhausted is archived at `eval/bench-64x64-paygo.json` (570s, 488 Tavily, also not submitted). Unique `(clue, pattern)` cache kept Tavily well under 4× the 32×32 search count.

langgraph-tf (`eval/bench-64x64.json`):

- 2,008,831 ms (**2009s**), 277 turns, submitted
- 17,659,112 TF tokens
- 98.8% letters / 97.4% words vs key (`solved: false` — first scale size where LangGraph also misses the generator key)
- **$30.90** TF

search-jev is **3.9×** wall, **4,908×** fewer TF tokens, **8.0×** cheaper full-stack. Neither agent is 100% vs the generator key at this size; Hermes is right that tiled 3×3s are a cost test. The 32-call TF leftover cap is why search-jev did not submit: Tavily/Jev filled most tiles, repair did not drain every leftover component.

| Size | minis | search-jev wall | LangGraph wall | search-jev TF | LangGraph TF |
|---|---:|---:|---:|---:|---:|
| 16×16 | 16 | 166s | 190s | 1.5k | 243k |
| 32×32 | 64 | 211s | 786s | 3.3k | 2.13M |
| 64×64 | 256 | 511s | 2009s | 3.6k | 17.7M |

LangGraph TF tokens roughly 8–9× per doubling of minis (history + get_state grow). search-jev TF stays in the low thousands because of the leftover cap; Tavily unique-clue cache is why 64×64 is 476 searches, not 4×211.

Race command:

```bash
LANGCHAIN_TRACING_V2=false PYTHONUNBUFFERED=1 \
  .venv/bin/python -u scripts/race_scale.py 64 new   # xword venv
LANGCHAIN_TRACING_V2=false PYTHONUNBUFFERED=1 \
  /Users/jkw/nebiusFDE/.venv/bin/python -u scripts/race_scale.py 64 old
```

`race_scale.py` sets `LANGCHAIN_TRACING_V2=false` after dotenv. Old max turns is `max(80, size*6)` so 64×64 is allowed 384 turns (32×32 finished in 82).

## How to read the take-home

1. **Routing tax.** LangGraph’s 13×13 demo was 98 turns / 996s / 2.44M tokens, mostly “which slot / are we done.” search-jev never spends a Token Factory turn on that.
2. **Grounding vs generation.** On interlocking 5×5s, Tavily + Jev + a Python grid solves the puzzle at 0 TF tokens.
3. **Scale is a throughput test.** Tiled 3×3s measure QPS and dollars, not “can it solve a Sunday.” Accuracy gold in this repo is `example_7x7` and a real 13×13, not `race_64x64`.
4. **The USD crossover.** Tavily basic search is $0.008. A 3×3 of generation tokens is cheaper than three searches. A 32×32 of generation tokens is not cheaper than 211 searches.

Reproduce: `python scripts/race_scale.py {16,32,64} {new,old,both}`.
