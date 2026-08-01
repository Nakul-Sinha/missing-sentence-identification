# Meridian Ashes - Missing Sentence Identification

Offline, from-scratch solution for the Meridian Ashes challenge.

## Result

The selected model achieved a grouped validation MeridianScore of **0.79808**:

| Component | Score |
| --- | ---: |
| Exact binding | 0.72647 |
| Membership | 0.98632 |
| Position | 0.81604 |
| Precedence | 0.84178 |

Rows sharing a long exact context sentence are placed in the same split, so
near-duplicate story excerpts cannot cross from training into validation.

## Model

`solution.py` trains everything from random initialization on `train.csv`:

1. A byte-BPE tokenizer is fitted on training text only.
2. A compact GPT-style causal language model is trained on reconstructed
   original excerpts in both forward and reverse token order.
3. Each candidate insertion receives separate whole-sentence fluency and local
   boundary-coherence scores from both directions.
4. A supervised coherence verifier fine-tunes the final two causal blocks to
   assign the three same-token candidate pairs bijectively to the three gaps.
5. The decoder evaluates the 48 valid hypotheses: one member from each
   same-token original/counterfeit pair, assigned bijectively to three gaps.

There are no pretrained weights, external corpora, TF-IDF features, hosted
inference calls, test-time fitting steps, package installers, or network calls.

The design follows the local/global coherence and constrained-ordering ideas in
[Shen and Baldwin (2021)](https://aclanthology.org/2021.sigdial-1.16/) and
[Oh et al. (2019)](https://aclanthology.org/D19-1232/), adapted to train from
scratch on the supplied challenge data.

## Run

The platform supplies the public-data directory and submission output path as
two positional arguments. Run:

```bash
python3 solution.py <public_dir> <submission_out>
```

The script reads `<public_dir>/train.csv` and `<public_dir>/test.csv`, creates
the parent directory for `<submission_out>`, and writes the CSV exactly there.

The selected validation run completed on a Kaggle T4 in 3,777 seconds
(approximately 63.0 minutes), including four verifier epochs used to select the
best checkpoint. The production defaults use the selected two verifier epochs;
the final all-training-data run, including test scoring and CSV creation,
completed in 3,479 seconds (approximately 58.0 minutes).

For the leakage-safe validation used above:

```bash
python solution.py dataset/public working/validation.csv --mode validate --ensemble 1
```
