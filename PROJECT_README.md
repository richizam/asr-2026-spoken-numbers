# ASR-2026 Spoken Numbers Challenge — Reference Implementation

Russian spoken-numbers ASR under a **5 M parameter** budget, trained **from scratch** for the Kaggle challenge
[`asr-2026-spoken-numbers-recognition-challenge`](https://www.kaggle.com/competitions/asr-2026-spoken-numbers-recognition-challenge).

## What this is

- **Char-CTC Conformer** (~3.1 M params) predicts Russian number words, a trie-snap post-processor maps them back to an integer in `[1 000, 999 999]`.
- Trained on 12 553 utterances from 6 speakers; validated on 10 speakers (4 out-of-domain); evaluated on 14 Kaggle test speakers.
- Heavy augmentation pipeline (MP3 re-encoding, MUSAN noise, reverb, pitch/VTLP, speed, SpecAugment) for OOD-speaker robustness — the winning lever on this task.

## Layout

```
src/
  text/          int <-> Russian words <-> CTC labels (+ trie snap)
  data/          CSV-driven Dataset, augmentations, speaker-balanced sampler
  models/        LogMelFilterBanks + Conformer-CTC small
  decode/        Greedy / prefix-beam / KenLM shallow fusion / rescoring
  lm/            KenLM corpus generator
  train.py       Resumable training loop (fp16, AdamW, cosine)
  infer.py       Produces submission.csv (ensembles via logit averaging)
scripts/
  prepare_data.py     Resample all audio to 16 kHz .npy shards
configs/
  conformer_ctc.yaml  Single source of truth for hyperparameters
tests/                Pytest unit tests (text roundtrip, model shapes, decoder)
kaggle/
  submission_notebook.ipynb   Kaggle notebook that loads weights from a GitHub release
report/
  report.md         Skeleton for the Google Classroom PDF deliverable
```

## Install

```bash
pip install -r requirements.txt
# KenLM (optional, for LM fusion) — build from source:
pip install https://github.com/kpu/kenlm/archive/master.zip
```

## End-to-end usage

### 1. Pre-cache audio (one-time)

```bash
python scripts/prepare_data.py \
    --data-root asr-2026-spoken-numbers-recognition-challenge \
    --cache-root cache_16k
```

### 2. (Optional) download augmentation corpora

- MUSAN: https://www.openslr.org/17/
- Room impulse responses (RIRS_NOISES): https://www.openslr.org/28/

Set their paths in `configs/conformer_ctc.yaml` under `data.noise_dir` / `data.rir_dir`.

### 3. Train

```bash
python -m src.train --config configs/conformer_ctc.yaml
```

Checkpoints go to `runs/conformer_ctc_small/{last,best}.ckpt`. Training is
resumable — restart the same command and it picks up from `last.ckpt`.

### 4. Build a Russian KenLM (optional, for shallow fusion)

```bash
python -m src.lm.build_lm --out-corpus lm_corpus.txt
lmplz -o 3 --text lm_corpus.txt --arpa lm.arpa
```

Set `decode.lm_path: lm.arpa` in the config.

### 5. Generate a submission

Single model:

```bash
python -m src.infer \
    --config configs/conformer_ctc.yaml \
    --ckpt runs/conformer_ctc_small/best.ckpt \
    --test-csv asr-2026-spoken-numbers-recognition-challenge/test.csv \
    --data-root asr-2026-spoken-numbers-recognition-challenge \
    --out submission.csv
```

Ensemble (logit averaging):

```bash
python -m src.infer --config configs/conformer_ctc.yaml \
    --ckpt runA/best.ckpt runB/best.ckpt runC/best.ckpt \
    --test-csv .../test.csv --data-root ... --out submission.csv
```

### 6. Submit via Kaggle notebook

Edit `kaggle/submission_notebook.ipynb`:

- Set `GITHUB_REPO_URL` to your public repo.
- Set `WEIGHTS_URL` / `LM_URL` to your GitHub release assets.

Upload the notebook to the Kaggle competition page and run.

## Testing

```bash
pytest -q                    # runs all unit tests
pytest tests/test_text_roundtrip.py -v   # text module (roundtrip on 10k ints)
pytest tests/test_model.py -s            # model forward/backward + param count
pytest tests/test_decoder.py -v          # greedy/beam/snap correctness
```

## Verification checklist before submission

- [ ] `pytest -q` passes.
- [ ] `python -m src.train --config ...` completes at least one epoch with dev CER logged per speaker.
- [ ] Dev harmonic-mean CER is reasonable (< 0.3 after a few epochs is normal; converge to < 0.1).
- [ ] `submission.csv` has header `filename,transcription`, all `transcription ∈ [1000, 999999]`, row count == test set size (2582).
- [ ] GitHub release contains `best.ckpt` (+ optional `lm.arpa`).
- [ ] Kaggle notebook runs end-to-end from a fresh kernel and writes `submission.csv`.

## Publishing to GitHub

```bash
git init
git add .
git commit -m "Initial commit: ASR-2026 Conformer-CTC baseline"
# Create an empty public repo at github.com first, then:
git remote add origin https://github.com/<user>/<repo>.git
git branch -M main
git push -u origin main

# After training, create a release and attach weights:
gh release create v1.0 runs/conformer_ctc_small/best.ckpt lm.arpa \
    --title "Weights v1.0" --notes "Dev CER X.XXX, leaderboard Y.YYY"
```

## Design decisions (short version)

| Decision | Why |
|---|---|
| Char-CTC over digit-CTC | CTC aligns naturally to phones, not to implicit "тысяча"→zeros. |
| Trie snap post-processor | Guarantees output ∈ `[1 000, 999 999]` even with noisy decoder. |
| Speaker-balanced sampler | `spk_E` is 45 % of train; uniform sampling dominates the gradient. |
| MP3 re-encoding augmentation | ~30 % of dev/test is MP3; lossy artifacts must be in the training distribution. |
| Pitch shift + VTLP | Simulates unseen vocal tracts. OOD CER is the tie-breaker metric. |
| Conformer, not Transformer | Local conv gives phoneme-level context that pure attention misses in small models. |
| Sinusoidal PE (not relative) | ~0.5 M param savings, fine at this data scale. |

See `report/report.md` for the full writeup template.
