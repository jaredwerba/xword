# Accuracy analysis (Hermes silo)

Written 2026-09-19. Do not edit `src/`, `scripts/`, or `data/puzzles/` from this note. Grok owns those. This file is the handoff.

Goal: explain what is being built, how, why search-jev is less accurate than LangGraph on the 16×16 race, and what to change if the metric that matters is matching the answer key.

## Direct answers

Is every fill 3 letters?
No. It depends which file you look at.

| Puzzle | Kind | Slot lengths | Connected? |
|---|---|---|---|
| example_3x3 | constructed mini | 6×3 | yes, one component |
| example_mini_5x5 | constructed mini | 4×3, 4×4, 2×5 | yes |
| example_5x5_b | constructed mini | 4×3, 4×4, 2×5 | yes |
| example_7x7 | constructed mini | 18×3, 4×7 | yes (pinwheel) |
| race_16x16 | generated tiles | **96×3** | **no** — 16 isolated 3×3s |
| race_32x32 | generated tiles | **384×3** | **no** — 64 isolated 3×3s |
| race_64x64 | generated tiles | **1536×3** | **no** — 256 isolated 3×3s |
| boatload-272971 (old repo) | real 13×13 | 13×3, 30×4, 5×5, 2×6, 8×8, 2×11 | yes |

The accuracy miss everyone is talking about (92% vs 100%) is on race_16x16, where every slot is three letters. The in-repo puzzles that actually look like crosswords (5×5, 7×7) were solved 100% on the small gold runs, and the 7×7 has never been live-scored for search-jev.

Is this a real crossword or a generated easy one?
Both exist. The take-home fixtures (3×3, two 5×5s, 7×7) are hand-built minis with real clue style and one interlocking grid. They are easy NYT-mini cousins, not a Sunday puzzle.

The 16/32/64 “scale” races are not real crosswords. `scripts/make_scale_puzzle.py` stamps independent 3×3 wordlist fills into a big board separated by streets of blocks. No theme, no 4+/5+/8+ answers, no constraint that crosses a tile boundary. ~190 distinct 3×3 wordlist fills showed up in a 200-seed sample — the structure almost never pins a unique fill. Clues are short dictionary glosses, plus at least one placeholder (`76A EVE` = “Three-letter fill (E__)”).

The old repo already has a real interlocking 13×13 (`nebiusFDE/data/external/boatload-272971.json`, 60 entries). That is the puzzle the original LangGraph agent was measured on. It has not been imported into `xword`.

If you keep racing tiled 3×3s you will keep measuring “can Tavily/Jev pick the generator’s synonym for a 3-letter word,” not “can the agent solve a crossword.”

## What is being built

New agent (`/Users/jkw/xword`, live www.jwerba.com/xword):

```
code picks slot (MRV)
  wordlist ∩ pattern          # 0 tokens
  Tavily search               # if empty or >12, or first letters
  Jev Choice(≤12 + none)      # ranker, not a solver
  Token Factory ≤8 guesses    # leftovers + a late repair pass
Python fill_slot rejects crossing conflicts
submit in code
```

Old agent (`/Users/jkw/nebiusFDE`): LangGraph + DeepSeek V4 Pro. The model sees the whole state and may fill several slots in one generation turn. Cost is the routing + generation. On 16×16: 190s, 242,739 TF tokens, 100% key match.

New agent on the same 16×16: 166s, 1,527 TF tokens, 91.7% letters / 87.5% words, grid complete, wrong key.

Small interlocking fixtures (recorded-offline.json): new agent 100% on 3×3 and mini 5×5 at 0 TF tokens. That is a different problem than the 16×16.

## How it is being built (the parts that hurt accuracy)

1. Slot-at-a-time, not grid-at-a-time.
   Jev’s state is `{slot, clue, pattern, source, options}`. It does not see the crossing clue, the rest of the 3×3, or the answer key’s other entries. LangGraph’s `get_state` does. On a 3-letter slot with an empty pattern, many legal crossword answers exist. The old model writes a consistent mini in one turn. The new model commits a locally plausible word and hopes crossings save it.

2. Tavily returns a valid crossword answer, not the keyed one.
   Concrete 16×16 traps (Grok already named two of these):
   - `10A SOB` clue “Crybaby” — WAH is also a 3-letter crossword fill.
   - `50A TOE` clue “Piggy” — PIG is the more famous answer (and is 3 letters; it just is not this slot).
   - `24A LOW` clue “Moo” — COW is the default crossword answer (7×7 uses “Moo maker” → COW).
   - `54A AXE` clue “Chopper” — HEL / AXE / ADZ all live in crossword-land.
   Once a synonym is written, `fill_slot` accepts it (no key check; that is correct for a solver) and the rest of the mini is now solving a different puzzle than the scorer.

3. Unique wordlist autofill with 2 letters, no clue check.
   `_run_region` / `_events`: if a slot has ≥2 letters and exactly one wordlist candidate, it is written without Jev and without the crossing clue. That is a speed trick. On 3-letter slots it is also how a wrong first fill poisons two crossings and then “proves” the wrong downs.

4. Undo is the oldest fill, not the contradiction.
   `fill_stack.pop(0)` — FIFO. When the mini is stuck, the first committed Tavily hit is undone even if the later unique-wordlist fill is the actual poison. Rejected pairs are never retried. There is no “this crossing’s clue is now impossible” detector.

5. Parallel regions (≥4 components) skip the serial event loop.
   `solve()`: if `len(regions) >= 4`, ThreadPool `_run_region` + a Token Factory `fill_region` repair. That is the 16×16/32×32 path. Repair only runs on leftover empties, not on complete-but-wrong minis. A fully filled wrong 3×3 is scored as done.

6. Global TF cap `tf_calls >= 32`.
   Fine on a 6-slot mini. On 96 slots / 16 regions it means most tiles never get a generation model, which is the only layer that can jointly satisfy six clues. LangGraph spent 20 generation turns and got the key. Search-jev spent 16 TF calls and 1,527 tokens and did not.

7. The scorer only knows one key.
   `eval/metrics.py` is letter/word match against `puzzle.solution`. A different valid 3×3 that answers the clues is a miss. On interlocking 5×5/7×7 that is usually the right metric (few alternate fills). On tiled 3×3s it over-punishes synonym fills. You want both numbers: key-match and “every slot is a clue-plausible word that crosses.”

8. Wordlist has no 6/8/11 letter words.
   1663 words: 392/697/499/75 of length 3/4/5/7. A real 13×13 (boatload) cannot be filled from this list. Scale-up therefore invented tiles instead of importing the real puzzle. That choice made the accuracy problem look like a Tavily ranking problem.

## Why it is less accurate (one paragraph)

LangGraph is slow because it spends a generation model on the whole mini and writes a jointly consistent fill. Search-jev is cheap because it treats each 3-letter clue as an independent web lookup plus a ranker that cannot see crossings. On an interlocking 5×5, longer answers and crossings still pin the key, so it looks perfect. On 16 isolated 3×3s, Tavily/Jev pick a legal synonym, unique-wordlist locks the rest, undo cannot surgically retract, repair never touches a complete wrong tile, and the scorer marks 8% of letters wrong. The 16×16 is also the wrong exam: it is 16 easy minis with no long answers, generated so the wordlist could fill them.

## What to change (accuracy only; Grok implements)

Do these in order. Do not keep scaling 32×32/64×64 as an accuracy test.

1. Stop using tiled 3×3s as the accuracy gold. Keep them as a cost/QPS stress if you want. Accuracy gold, in this repo, should be:
   - example_7x7 (already here, 4 seven-letter answers, one component)
   - example_5x5_b (trickier clues; no live search-jev row yet)
   - boatload-272971 imported as an xword fixture (real 13×13, mixed lengths)

2. Put crossings in Jev’s state. Minimum: this slot’s clue+pattern, each crossing’s clue+pattern, letters already placed. Rank with that, not a naked 3-letter gloss.

3. Do not autofill a unique wordlist hit until the crossing clues still have ≥1 Tavily/wordlist candidate each. If a down clue becomes impossible, reject the across that caused it.

4. Undo the most recently unconstrained fill (LIFO) or the fill that most reduced crossing candidate-counts, not `pop(0)`.

5. When a component is 6 slots of length 3, one Token Factory `fill_region` up front (what LangGraph does) will beat 6 Tavily+Jev calls on key-match. Use Tavily to ground the prompt, not to commit the first letter.

6. Report two scores: `letter_accuracy` vs key, and `clue_valid` (optional human/LLM check, or “word in wordlist AND crossings legal”). The 92% number is key-match on a generator that admitted many fills.

7. Live-score search-jev on example_7x7 before claiming the new loop is “less accurate.” It may already be fine on interlocking minis. The 16×16 result does not prove otherwise.

## Files read (no writes outside this folder)

- src/xword/agent.py (committed + Grok’s uncommitted parallel/repair path, read-only)
- src/xword/grid.py, solver.py, eval/metrics.py
- data/puzzles/*.json, eval/bench-16x16.json, eval/recorded-offline.json
- scripts/make_scale_puzzle.py
- nebiusFDE/data/external/boatload-272971.json
- Grok session notes on 16×16 (WAH vs SOB, PIG vs TOE)

No Tavily/Jev/TF calls from Hermes. Grok’s 32×32 race was left alone.
