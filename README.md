# Optimizer-equipped CNN-MLP on CIC-DDoS2019

A replication and adaptation of

> Mehmood S., Amin R., Mustafa J., Hussain M., Alsubaei F.S., Zakaria M.D. (2025).
> *Distributed Denial of Services (DDoS) attack detection in SDN using
> Optimizer-equipped CNN-MLP.* PLOS ONE 20(1): e0312425.

**Not a bit-for-bit reproduction, and it does not claim to be.** The paper leaves
most of its experimental protocol unstated and contradicts itself on the results.
Every gap we filled and every disagreement we found is written down here and in
`config/run_config.json`, so the thesis can defend each choice rather than assert it.

The detailed source analysis lives in two Vietnamese documents beside this repo:
`SPEC_Cu_the_hoa_bai_bao.md` (what the paper says, omits, and contradicts) and
`DESIGN_Buoc1_Kien_truc.md` (architecture and run matrix).

---

## What the paper does not say

Grepping the full 29 pages: these never appear.

| Missing | Consequence |
|---|---|
| Learning rate | The string `0.001` appears **zero times**. Our `lr=0.001` is a user constraint, recorded as `learning_rate_source: user_constraint`, **not** as "matching the paper" |
| Batch size, epoch count | Ours are constraints: 4096 and 100 |
| Train/val/test split | The paper describes no split at all. Every protocol decision here is ours |
| How the two branches join | It draws each branch and never the junction. Late concatenation is our design |
| Dropout | Never mentioned. We add it and the Bayesian search tunes it |
| Which SHAP explainer | Never named |
| Bayesian search space, budget, acquisition | Never given |

## Three findings that changed the design

**The paper's SHAP tables are single-instance waterfalls, not summary plots.**
Table 1 and Table 2 are captioned "Summary plot", but their values sum to
`f(x) − E[f(x)]`: +8.960 against +8.961 for Table 2, matching to three decimals.
That is the local-additivity identity, so they describe one record each, not a
global ranking. A `mean(|SHAP|)` ranking is therefore **not comparable** to them.
We emit both: the global ranking that drives selection, and a waterfall built the
same way for an honest comparison.

**The model has 40 inputs, not 20.** Both tables show 19 named features plus a row
reading "21 other features", so the explained model saw 40. `top_k = 40`.
How the paper went from 88 raw features to those 40 is never described.

**The paper up-samples before feature selection, and never mentions splitting.**
Section 2.5: *"The disbalance of the data distribution can be solved by using
up-sampling to make a balanced dataset"*, placed before feature selection in Fig 6.
Duplicating BENIGN rows and only then dividing the data puts copies of the same
record on both sides. On a dataset that is **0.16% BENIGN**, that is the most
likely explanation for 99.95%. The `paperlike` variant reproduces this deliberately;
the delta against `main` is this thesis's estimate of how much is leakage.

## The paper contradicts itself on its own results

| Source | Accuracy |
|---|---|
| Abstract, §3 p.18, Conclusion p.26 | **99.95%** |
| §3, final paragraph, p.26 | **95%** |
| Fig 11, p.23 | 99.9% |

It also prints F1 as `0.994%` — a percent sign on a value already in [0, 1] — and
glosses its own "OPT" as "Optimal Transport Problem" (p.23) when everywhere else
it means the Adam + Bayesian optimizer.

`comparison_with_paper.csv` therefore carries **two** reference columns,
`paper_headline` and `paper_body`. Collapsing them to one would mean choosing
which figure to be judged against, and the flattering choice is the tempting one.

---

## Deviations from the paper

| Item | Paper | Here | Why |
|---|---|---|---|
| Convolution | Conv2D 3×3, MaxPool 2×2 | **Conv1D** k=3, MaxPool1d(2) | The input is a 1-D flow record. `conv2d_reshape` is kept as an ablation and flags its own reshape artefact |
| Branch fusion | not described | late concatenation | Filling a gap, not reproducing a choice |
| Split | none stated | temporal, stratified by (file, label) | See below |
| Up-sampling | before feature selection | **not in the main run**; reproduced in `paperlike` | Isolates the variable and measures the leak |
| Framework | implied Keras | PyTorch CPU | Exact checkpoint and RNG control for resume |
| Features | 40 (inferred) | 40 (`main`), all (`featall`), 20 (`feat20`) | Sensitivity rather than assertion |
| Metrics | Acc / P / R / F1 | + Balanced Acc, MCC, ROC-AUC, PR-AUC, Log Loss, specificity, FPR, FNR | Accuracy is meaningless at 1:618 — see below |
| Hardware | i7, 8 GB, AMD RX 550 | Kaggle CPU | **Equivalent**: an RX 550 has no CUDA, so the paper also trained on CPU |

### Why the split is stratified by label

BENIGN is clustered in time, not spread through the captures: 34,822 of the
35,790 BENIGN rows in `03-11/Syn.parquet` sit at the tail, and 20,866 of 25,247
do the same in `01-12/TFTP.parquet`. Cutting each file as one time-ordered stream
leaves validation with ~2,900 BENIGN rows out of 8.5M, far too thin to optimise
Macro-F1 on. Cutting each *(file, label)* stream separately keeps every family
and both classes present in the right proportion.

The cost, stated plainly: test is no longer strictly later in wall-clock time
than train across the dataset. Within each (file, label) stream it still is, so
no burst straddles a boundary — which is what actually blocks leakage.

### Why training is subsampled but test is not

70.4M rows × 100 epochs does not fit, and the float32 cache is 21.8 GiB against a
12 GiB budget. So **train and validation** keep every one of the 113,828 BENIGN
rows and thin attacks to 1:30. **Test is untouched**, keeping the natural 1:618
prior — it is a single forward pass, so its size costs little, and the headline
numbers are then measured at the real operating point with no prior-shift caveat.

### Why not accuracy

At the test prior, predicting "attack" for every row scores **99.84% accuracy**.
`test_metrics.json` reports that number as `accuracy_of_always_attack` next to
the model's own, so the comparison is unavoidable.

---

## Layout

```
configs/          experiment, dataset, preprocessing, model, bayesopt
  variants/       the seven runs; each holds only what it changes
src/              pipeline modules, no plotting outside viz.py
scripts/          manifest minting, reporting, profiling, acceptance checks
tests/            170 tests
.github/workflows/run-kaggle.yml
```

## Pipeline

```
audit -> split -> preprocess -> SHAP select -> Bayesian search -> train 100 -> evaluate -> explain
```

Order is load-bearing. Deduplication happens **before** the split; the scaler,
imputer, SHAP selector and Bayesian search are all fitted on **train only**; test
is read exactly once, by `evaluate.py`.

## Running

```bash
pip install -r requirements.txt          # torch from the CPU index
python -m pytest                          # 170 tests

python src/variants.py --variant main --out work/experiment.yaml
python src/data_audit.py    --dataset cicddos2019 --input-root DATA --out-dir work/config
python src/split.py         --input-root DATA --out-dir work/config --experiment-config work/experiment.yaml
python src/preprocessing.py --input-root DATA --split-dir work/config --out-dir work
python src/shap_selection.py --cache-dir work/cache --out-dir work
python src/bayesopt.py       --cache-dir work/cache --out-dir work
python src/train.py          --cache-dir work/cache --out-dir work --run-id RUN --best-params work/bayesopt/best_params.json
python src/evaluate.py       --cache-dir work/cache --out-dir work
python src/explainability.py --cache-dir work/cache --out-dir work

python scripts/make_report.py       --run-dir work
python scripts/validate_artifacts.py --run-dir work
```

On Kaggle none of this is typed by hand: GitHub Actions mints presigned URLs and
`src/session.py` drives the phases.

## Security

The Kaggle notebook holds **no AWS credentials** and never calls boto3. Actions is
the only place the repository secrets are read; it signs short-lived URLs and hands
the notebook one URL pointing at a manifest. URLs travel by file, never on a command
line. Every log line is redacted, and a test asserts the committed notebook contains
no signing material.

## Resume

A Kaggle CPU session is capped at 12 hours and can be cancelled sooner, so a
100-epoch run spans several. Work is checkpointed by epoch; cancellation by the
time limit exits 0 with `status: RESUME_REQUIRED`, which is a normal outcome and
not a failure.

`test_interrupted_run_matches_an_uninterrupted_one` runs training twice — once
straight through, once cut off after every epoch — and requires identical weights
to `1e-6`. Restoring the optimiser but not the RNG produces well-formed artifacts,
a complete history, and a different model; only comparing weights catches it.

## What this repository will not tell you

- **SHAP does not show causation.** A high value says the model leaned on that
  feature, not that the feature causes an attack.
- **Port and protocol importances reflect the capture setup** as much as attack
  behaviour, and do not transfer to another network.
- **`paperlike` metrics are invalid by construction.** They exist to be subtracted.
- **A result below 99.9% is not a failed replication.** It is the expected
  consequence of removing leakage the paper did not address, and analysing that
  gap is the contribution — not something to hide.
