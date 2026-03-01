# Max Toilet Event Detection Pipeline

Automatically classifies Tapo video clips as: **wee**, **poo**, or **neither**.

---

## Setup

```bash
pip install ultralytics scikit-learn joblib opencv-python numpy pandas
```

YOLOv8 weights (~6 MB each) download automatically on first run.

---

## The Four Stages

### Stage 1 — Mine (reduce 1000s of clips to ~100–200)

```bash
python3 stage1_miner.py --clips /path/to/tapo/clips --out shortlist.csv
```

Options:
- `--top 200`       how many clips to shortlist (default 200)
- `--min-score 10`  minimum score to include
- `--no-yolo`       skip dog detection (faster, less accurate)

Output: `shortlist.csv` ranked by toilet-event likelihood.

---

### Stage 2 — Review (build your ground truth, manually)

```bash
python3 stage2_reviewer.py --shortlist shortlist.csv --labelled ./labelled
```

Controls during review:
- `w` — wee
- `p` — poo  
- `n` — neither
- `s` — skip (review later)
- `q` — quit (saves progress, resumable)

Output:
- `labels.csv` — your decisions
- `labelled/wee/`, `labelled/poo/`, `labelled/neither/` — sorted clips

**Aim for at least 20–30 clips per class before Stage 3. More is better.**

---

### Stage 3 — Train (uses your labels to build a real model)

```bash
python3 stage3_trainer.py \
  --labelled ./labelled \
  --labels labels.csv \
  --model dog_toilet_model.joblib
```

Features it learns from:
- **Hip height** — low bbox bottom = squatting posture
- **Spine angle** — horizontal = crouching
- **Dwell time** — staying still in one spot
- **Motion pattern** — approach → still → leave
- **Bbox aspect ratio** — wide+short = squat

Feature extraction is cached in `features_cache.json` — re-running is fast.

Output: `dog_toilet_model.joblib`

---

### Stage 4 — Classify (run on new clips)

```bash
# Single clip
python3 stage4_classifier.py --model dog_toilet_model.joblib --clip new_clip.mp4

# Whole folder
python3 stage4_classifier.py --model dog_toilet_model.joblib --clips /new/clips --out results.csv

# Watch mode (checks every 10 seconds for new clips)
python3 stage4_classifier.py --model dog_toilet_model.joblib --watch /tapo/clips --out results.csv
```

Output: label, confidence score, and timestamp of the event within the clip.

---

## Workflow

```
Your Tapo clips
     │
     ▼
stage1_miner.py ──────────► shortlist.csv (top ~200 clips)
     │
     ▼
stage2_reviewer.py ────────► labels.csv + labelled/ folders
     │
     ▼
stage3_trainer.py ─────────► dog_toilet_model.joblib
     │
     ▼
stage4_classifier.py ──────► results.csv (wee / poo / neither + timestamp)
```

---

## Improving the model over time

1. Run Stage 4 on new clips
2. Find misclassified ones (check low-confidence results first)
3. Add them to `labels.csv` with correct labels
4. Re-run Stage 3 to retrain
5. Model gets better with each iteration

---

## Notes

- The current heuristics (Stage 1) deliberately have **high recall** — better to review a false positive than miss a real event
- Stage 3 learns Max's specific posture and behaviour from *your* labelled data, so it adapts to your camera angle and Max's habits
- If accuracy plateaus, consider YOLOv8-pose with dog-specific keypoint training (future work)
