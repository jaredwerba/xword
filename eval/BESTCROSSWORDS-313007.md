# Stick With It! — v1 vs v2

Puzzle: Sam Brody, BestCrosswords 313007, 15×15 American, 80 clues, 19 Sep 2026.  
Play: https://www.bestcrosswords.com/themed-crossword-puzzles/313007-stick-with-it  
Theme: treats on a stick.

Official solution is next-day. **v1’s fill is the reference**: complete, crossing-legal, and it hits MAGI / POPSICLE / CARAMELAPPLE / CORNDOG / COTTONCANDY / LOLLIPOP / TRUETO / ONTAP / SPACE.

Current v2 loop: wordlist + Jev on tight patterns, Token Factory `fill_region` for leftovers, Tavily last. Agent `search-jev-region` in `eval/bench-313007.json`.

| | **v1 LangGraph** | **v2 fill_region loop** | v2 old (Tavily-first) |
|---|---:|---:|---:|
| Wall | 335s | **29s** | 366s |
| Submitted | yes | **yes** | no |
| Letters filled | 186/186 | **186/186** | 185/186 |
| vs reference letters | 100% | **98.9%** (184/186) | 86.6% |
| TF tokens | 517,268 | **2,484** | 4,094 |
| Tavily | 0 | **0** | 87 × $0.008 |
| Jev | 0 | 1 | 77 |
| **USD** | $0.91 | **$0.004** | $0.70 |

v2 misses vs v1: `TREETO` vs `TRUETO`, `MORELE` vs `MORALE`. Theme is complete: POPSICLE, CARAMELAPPLE, CORNDOG, COTTONCANDY, LOLLIPOP.

Old Tavily-first v2 near-misses (poisoned first letter): MAGI→MBHT, POPSICLE→PARLICLE, CARAMELAPPLE→NARAMELAPPLE, CORNDOG→MORODOG, LOLLIPOP→LALLIP.O. The fill_region loop does not do that.

Reproduce: `data/puzzles/bestcrosswords-313007.json`, benches in `eval/bench-313007.json`.

## Ablation: Tavily off

`search-jev-no-tavily` on the same puzzle (empty search, same TF cap 32, 250 steps):

| | v1 | v2 + Tavily | v2 no Tavily |
|---|---:|---:|---:|
| Wall | 335s | 366s | **80s** |
| Letters filled | 186/186 | 185/186 | **12/186 (6.5%)** |
| vs reference | 100% | 86.6% | **4.3%** |
| USD | $0.91 | $0.70 ($0.696 Tavily) | **$0.009** |

Tavily is the fill engine, not an optional extra. The wordlist is only 3/4/5/7 letters, so 6/8/11/12-letter theme answers cannot come from TF leftovers (`guess_words` is filtered to the wordlist). No Tavily ⇒ no POPSICLE / CARAMELAPPLE / COTTONCANDY.

Jev on the no-Tavily run: **34 decisions, 0 reranks**. 23× `none` on Token Factory shortlists, 5× keep first, 2× `none` on wordlist. It gated leftovers; it did not pick the crossword. Jev USD **$0.0002**.

On interlocking 5×5 *with* Tavily, Jev did rerank after 1–2 letters (FIRE→TIRE, SUB→SUN). That job never fires here without a grounded shortlist.

