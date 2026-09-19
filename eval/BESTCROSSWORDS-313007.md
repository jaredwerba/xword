# Stick With It! — v1 vs v2

Puzzle: Sam Brody, BestCrosswords 313007, 15×15 American, 80 clues, 19 Sep 2026.  
Play: https://www.bestcrosswords.com/themed-crossword-puzzles/313007-stick-with-it  
Theme: treats on a stick.

Official solution is next-day. **v1’s fill is the reference**: complete, crossing-legal, and it hits MAGI / POPSICLE / CARAMELAPPLE / CORNDOG / COTTONCANDY / LOLLIPOP / TRUETO / ONTAP / SPACE.

| | **v1 LangGraph** (Nebius-XWord) | **v2 search-jev** (xword) |
|---|---:|---:|
| Wall | **335s** | 366s |
| Submitted | yes | no (1 empty cell) |
| Letters filled | **186/186 (100%)** | 185/186 (99.5%) |
| Words filled | **80/80** | 78/80 |
| vs reference letters | **100%** | **86.6%** (161/186) |
| vs reference words | **100%** | **68.8%** (55/80) |
| TF tokens | 517,268 | **4,094** |
| Tavily | 0 | 87 × $0.008 |
| Jev input | 0 | 43k |
| **USD** | **$0.91** | **$0.70** |

Theme (v1 all five; v2 only COTTONCANDY): POPSICLE, CARAMELAPPLE, CORNDOG, COTTONCANDY, LOLLIPOP.

v2 near-misses from a poisoned first letter: MAGI→MBHT, POPSICLE→PARLICLE, CARAMELAPPLE→NARAMELAPPLE, CORNDOG→MORODOG, LOLLIPOP→LALLIP.O. Same failure mode as the tiled 16×16: slot-by-slot Tavily/Jev commits, crossings lock, FIFO undo does not repair the mini.

v1 writes a whole region per generation turn (24 turns). That is slower-USD on TF and slightly faster wall here, and it is the accurate solve.

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

