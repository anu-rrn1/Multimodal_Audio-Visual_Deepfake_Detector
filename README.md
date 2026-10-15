# AVFF-v3: Multimodal Audio-Visual Deepfake Detection on FakeAVCeleb

An AVFF-based audio-visual deepfake detector evaluated on the full **FakeAVCeleb v1.2** dataset
(21,544 videos, 500 identities). The model learns audio-visual correspondence from real videos
only, then uses cross-modal inconsistency to detect manipulation in either the video track, the
audio track, or both.

**Keywords:** deepfake detection, multimodal learning, audio-visual fusion, self-supervised
learning, contrastive learning, cross-modal translation, transfer learning, ResNet-18, MFCC,
class imbalance, PyTorch, OpenCV

## Results

Held-out test identities (4,214 fake and 100 real videos, natural class distribution):

| Accuracy | Macro-Precision | Macro-Recall | Macro-F1 | AUC |
|:---:|:---:|:---:|:---:|:---:|
| 99.6% | 0.936 | 0.978 | 0.956 | 0.999 |

On a class-balanced view of the same predictions: accuracy 97.8%, macro-precision 0.979,
macro-recall 0.978, macro-F1 0.978.

Full metrics are in `results_ft/results.json`; per-video predictions are in
`results_ft/test_predictions.csv`.

## Method

**Dataset.** FakeAVCeleb v1.2 has four categories: RealVideo-RealAudio, RealVideo-FakeAudio,
FakeVideo-RealAudio and FakeVideo-FakeAudio. Labels are binary (real / fake), with the four
categories used for balanced sampling and auxiliary supervision.

**Features**
- *Video:* 16 uniformly sampled frames per video, face-cropped, 224x224. A ResNet-18 (layer3 and
  layer4 fine-tuned to separate genuine from manipulated face frames) gives per-frame features
  (layer4 average + max pooling, layer3 average pooling). Temporal-difference tokens expose
  frame-to-frame artefacts.
- *Audio:* 40 MFCCs per video-aligned slice (mean, standard deviation, delta, log-RMS energy;
  121-d), so audio and video tokens are time-aligned.

**Stage 1: self-supervised pre-training (real videos only)**
- *Complementary masking:* for each clip, half of the time steps keep video visible and audio
  hidden; the other half do the reverse.
- *Audio-video cross-modal translation (A2V / V2A):* transformer translators predict the hidden
  modality from the visible one, learning how genuine audio and video correspond.
- *Contrastive alignment:* a symmetric InfoNCE loss aligns audio and video embeddings of the same
  clip in a shared space, giving a lip-sync consistency score.

**Stage 2: supervised multi-task classification**
- Cross-modal reconstruction error, cosine similarity and sync score are used as inconsistency
  features, fused with the video and audio encodings by cross-attention.
- Heads: real/fake (main), plus auxiliary "video manipulated?" and "audio manipulated?" heads.
- Category-aware balanced sampling (50% real, fake half split across the three fake types),
  temporal-window augmentation (9 windows of 8 frames per video), label smoothing, AdamW with
  warm-up and cosine schedule, early stopping on validation, and 3-window test-time averaging.

**Evaluation protocol (leakage-free)**

| Step | Data used |
|---|---|
| Identity-grouped split | 80% development / 20% test; no person appears in more than one set |
| Backbone fine-tuning | training identities only; epoch chosen on validation |
| Stage 1 and Stage 2 training | training split |
| Early stopping and threshold | validation split |
| Final test | threshold frozen, evaluated once |

The decision threshold is the one that maximises validation Macro-F1 subject to accuracy,
macro-precision, macro-recall and macro-F1 all being at least 0.80. The backbone checkpoint stores
the identities it was trained on, and the training script refuses to run any fold whose
validation or test identities overlap them.

## Repository structure

| File | Purpose |
|---|---|
| `avff_v3_full.py` | Feature extraction, AVFF model, Stage 1 / Stage 2 training, evaluation |
| `finetune_backbone.py` | Fine-tunes the ResNet-18 on real vs. manipulated face frames |
| `eval_utils.py` | Identity-grouped splits, threshold selection, metrics |
| `RUN_GUIDE.md` | Step-by-step run instructions and timings |
| `results_ft/` | Final results: metrics, predictions, test split, training log |

## Usage

```bash
pip install -r requirements.txt
```

Download FakeAVCeleb v1.2 and the ImageNet ResNet-18 weights (`resnet18-f37072fd.pth`). Place
them next to this repo (or inside it), or point to them explicitly:

```bash
# Windows PowerShell
$env:FAKEAVCELEB_PATH = "D:\data\FakeAVCeleb_v1.2"
$env:RESNET18_WEIGHTS = "D:\weights\resnet18-f37072fd.pth"

# Linux / macOS
export FAKEAVCELEB_PATH=/data/FakeAVCeleb_v1.2
export RESNET18_WEIGHTS=/weights/resnet18-f37072fd.pth
```

The paths can also be passed with `--dataset_path` and `--weights_path`.

```bash
# 1. Fine-tune the visual backbone
python finetune_backbone.py

# 2. Extract features (resumable)
python avff_v3_full.py --mode extract --backbone_ckpt backbone_ft/resnet18_ft_fold1.pth \
    --audio_from features_v3 --cache_dir features_ft

# 3. Train and evaluate
python avff_v3_full.py --mode train --folds 1 --backbone_ckpt backbone_ft/resnet18_ft_fold1.pth \
    --cache_dir features_ft --out_dir results_ft
```

See `RUN_GUIDE.md` for details.

## Reproducibility

FakeAVCeleb ships without an official train/test split. The identity-grouped split used here is
fixed by `--seed 42` and recorded in `results_ft/test_split.json`.

## References

- Oorloff, J., Yacoob, Y., Shrivastava, A. *Avoid-Reconstruct-Fuse (AVFF): Self-Supervised
  Audio-Visual Representation Learning for Deepfake Detection.* CVPR 2024. 
- Khalid, H., Tariq, S., Kim, M., Woo, S. S. *FakeAVCeleb: A Novel Audio-Video Multimodal
  Deepfake Dataset.* NeurIPS Datasets and Benchmarks 2021. 
- He, K., Zhang, X., Ren, S., Sun, J. *Deep Residual Learning for Image Recognition.* CVPR 2016.
