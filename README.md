# Missing Sentence Identification

## The problem

I get a story excerpt with three sentences removed, plus a pool of candidate
sentences. Each real sentence is paired with a counterfeit built from the same
tokens, so I cannot lean on vocabulary overlap. I have to decide which candidates
are genuine and assign them bijectively to the three gaps. Scoring breaks into
exact binding, membership, position, and precedence.

## What I did

Everything trains from random initialization on the supplied training file. No
pretrained weights, no external corpora, no network calls, no TF-IDF.

I fit a byte-BPE tokenizer on the training text, then train a compact GPT-style
causal language model on reconstructed excerpts in both forward and reverse token
order. Reading in both directions matters, because a sentence that belongs in a
gap should look fluent from the left and from the right. Each candidate insertion
then gets separate whole-sentence fluency and local boundary-coherence scores from
both directions. A supervised coherence verifier fine-tunes the last two causal
blocks to handle the bijective assignment, and the decoder evaluates all 48 valid
hypotheses: one member of each original and counterfeit pair, assigned across the
three gaps.

Grouped validation score: 0.798, with exact binding at 0.726 and membership at
0.986.

## Layout

`python3 solution.py <public_dir> <submission_out>` runs it end to end.
`TECHNICAL.md` has the full method and the references the design draws on.
Datasets are not committed.
