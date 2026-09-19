# Is Jev beneficial here?

Hermes, 2026-09-19. Opinion, not a code change.

Short answer: yes as a ranker/abstainer on a grid-legal shortlist. No as a crossword solver. The 16×16 miss is not a reason to delete it.

## What Jev is for

System One: Choice / Noul over state you already have. Fast, cheap, typed, no token essay. It cannot invent first letters (Lamb Chops already proved HID > ATE/EAT). The README is right: “Jev is not a crossword solver.”

In this repo it is used as:
- Noul: does this one word fit the clue? (threshold 0.75)
- Choice: which of ≤12 options, or `none` (threshold 0.50)

That is the correct primitive. The bug is the state: `{slot, clue, pattern, options}` with no crossing clues.

## Keep it

1. Cost thesis of the new agent. Old loop spends DeepSeek on get_state / which-slot / submit. New loop needs a semantic gate between Tavily hits and fill_slot. Without Jev you either take Tavily[0] (worse) or call Token Factory to pick among 8 words (defeats cheaper-than-LangGraph). Jev is ~$0.042/MTok and hundreds of ms; V4 Pro is ~$1.75/MTok and seconds.

2. Abstention is the actual feature. `none` / low probability is what should send a slot to TF instead of writing WAH for “Crybaby.” A deterministic “first Tavily hit that matches the pattern” has no abstain. The 92% run still committed synonyms because Jev was asked “which of these answers the gloss?” not “which of these still leaves a legal down.” That is a question-design miss, not a model miss.

3. Interlocking minis already match the product. 3×3 and mini 5×5: 100%, 0 TF tokens, Jev did the ranking. That is the job. We have no live 7×7 number; do not throw Jev out before that run.

4. Take-home story. The Blueprint stack is Tavily (ground) / Jev (decide) / TF (leftover inference) / Python (grid). Strip Jev and the demo is “search + chat model,” which is a smaller claim than the old LangGraph agent plus a search box.

## Do not keep it for

- Jointly filling a 3×3 or a 13×13. That is TF `fill_region` or LangGraph.
- First-letter search. Already failed.
- Making tiled 3×3 races match a generator key. Crossings + longer slots do that; Jev ranking isolated glosses cannot.

## If you removed it tomorrow

Accuracy on race_16x16 would not jump to 100%. You would still write legal synonyms, still autofill unique wordlist hits, still FIFO-undo. You would lose the only cheap “this shortlist is junk, escalate” bit.

If you keep it, change the question: pass crossing clue+pattern, require the Choice winner to leave ≥1 candidate on every crossing, treat `none` as escalate-to-TF, never as “pick the next Tavily hit.” Then Jev is doing System One on a crossword, not playing dictionary.

Verdict: beneficial. Wrong seat on the 16×16 path. Do not rip it out to chase key-match on generated tiles.
