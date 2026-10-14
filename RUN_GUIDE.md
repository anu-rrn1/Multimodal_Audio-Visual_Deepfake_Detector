# AVFF v3: full-dataset run guide (CPU laptop)

## 1. Setup (once)
Put `avff_v3_full.py` and `eval_utils.py` in the same folder, then run:
```
pip install torch torchvision torchaudio opencv-python scikit-learn scipy imageio-ffmpeg
```
Weights: use the ImageNet file `resnet18-f37072fd.pth` (your `archive (16)` folder is auto-detected).
If it is not found, the script downloads it. Do **not** use an old fine-tuned `best_model.pth`,
because it may have been trained on videos that are now in the test set.

## 2. Extract features (one time, about 4–7 h, resumable)
```
python avff_v3_full.py --mode extract
```
* Shows videos/s and an ETA. Progress is saved every 512 videos to `features_v3\`.
  If the laptop sleeps or crashes, run the same command again and it continues where it stopped.
* Keep the laptop plugged in and set sleep to "Never".
* If it is too slow, use `--img_size 160` (about 2x faster, slightly weaker). If you change
  `--n_frames`, `--img_size` or `--face_crop`, also use a new `--cache_dir`.

## 3. Train + evaluate (about 30–60 min)
```
python avff_v3_full.py --mode train
```
Outputs go to `results_v3\`: `train_log.txt`, `results.json`, `test_predictions.csv`, and
`avff_v3_fold1..5.pth`.

Quick smoke test before the long run (debug only, not for reporting):
```
python avff_v3_full.py --mode all --max_samples 600 --epochs_s1 2 --epochs_s2 2 --cache_dir features_debug --out_dir results_debug
```

## 4. Protocol (leakage-free)
| Step | Data used |
|---|---|
| Identity-grouped split: 80% dev / 20% test | test identities never appear anywhere else |
| Stage 1 self-supervised (AVFF) | **real** videos of the fold's *train* split only |
| Stage 2 training | fold train split |
| Early stopping + threshold | that fold's validation split |
| Final test | 5-fold margin ensemble (each fold keeps its own validation threshold), evaluated **once** |

Threshold rule: on validation, keep the thresholds where Accuracy, Macro-Precision,
Macro-Recall and Macro-F1 are **all ≥ 0.80**, and pick the one with the highest Macro-F1.
If none qualifies, the log says so. Do **not** re-tune on test.

Options: `--thr_objective min_metric` picks the threshold with the largest safety margin
(often generalises better). `--thr_view class_balanced` optimises for a 50/50 deployment prior.

## 5. What to report
The script prints two views of the same test predictions:
* **natural**: the real 1:42 test distribution. Accuracy and fake-class scores are high almost
  automatically here; the macro and real-class numbers are the honest ones.
* **class_balanced**: each class carries equal weight. This equals the expected result on a
  50/50 test set, without random subsampling.

Always report AUC and AP as well, since they don't depend on the threshold.

Suggested Methods text:
> Videos were split by identity into a development set (80%) and a held-out test set (20%)
> using stratified group sampling, so that no subject appears in both. Five-fold
> identity-grouped cross-validation was performed on the development set. The self-supervised
> stage was trained only on real videos from each fold's training split. For each fold, the
> decision threshold was selected on that fold's validation split as the threshold maximizing
> Macro-F1 subject to Accuracy, Macro-Precision, Macro-Recall and Macro-F1 all being at least
> 0.80, and was then fixed. Test predictions of the five fold models were combined by averaging
> each model's logit margin relative to its own threshold, and the held-out test set was
> evaluated once.

If the validation step finds no threshold that reaches 0.80, write that the constraint
could not be satisfied and report the result as it is.

## 6. Fix 2: fine-tuned ResNet-18 (single split, about 9–11 h in total)
The ImageNet ResNet-18 has never seen deepfakes, so real videos and face-swapped videos with
real audio look alike to it. `finetune_backbone.py` fine-tunes layer3 and layer4 to separate
real-video frames (RealVideo-RealAudio and RealVideo-FakeAudio) from fake-video frames. The
video features are then extracted again with that backbone.

On a CPU this is affordable for **one** identity-grouped split, which is fold 1 of the same
split used above. Run the steps in this order:

| # | Command | Time |
|---|---|---|
| 1 | `python avff_v3_full.py --mode train --folds 1 --out_dir results_v31_fold1` (baseline: same split, ImageNet features) | ~30 min |
| 2 | `python finetune_backbone.py` (prints a GO / NO-GO line at the end) | ~2.5–4.5 h |
| 3 | `python avff_v3_full.py --mode extract --backbone_ckpt backbone_ft\resnet18_ft_fold1.pth --audio_from features_v3 --cache_dir features_ft` | ~5–5.5 h |
| 4 | `python avff_v3_full.py --mode train --folds 1 --backbone_ckpt backbone_ft\resnet18_ft_fold1.pth --cache_dir features_ft --out_dir results_ft` | ~30 min |

* Every step is resumable: run the same command again after a crash or sleep.
* Step 2 needs about 3.4 GB of disk for `frame_bank\`. You can delete that folder after step 2.
* If step 2 prints NO-GO, stop and look at the log before spending 5 h on step 3.
* Compare steps 1 and 4. Only the backbone differs, so this is a clean ablation.

Leakage protection: the backbone checkpoint lists every identity it was trained on. During
training, `avff_v3_full.py` stops with an error if any of them appears in the validation or test
set of the folds being run. For example, `--folds 2` with the fold-1 backbone is refused.

| Step | Data used |
|---|---|
| Backbone fine-tuning | fold-1 **train** identities only; epoch chosen on fold-1 validation |
| AVFF Stage 1 / Stage 2 | fold-1 train split |
| Early stopping + threshold | fold-1 validation split |
| Final test | single model, threshold frozen, evaluated **once** |

Caveat for the paper: the backbone has seen the training videos, so their video features are
more separable than those of unseen videos. The threshold and early stopping use unseen
validation identities, so the reported test numbers remain honest.

Suggested Methods text:
> To adapt the visual encoder to manipulation artefacts, the ImageNet ResNet-18 was fine-tuned
> (layer3, layer4 and a linear head) to classify individual face frames as genuine or
> manipulated video, using only identities of the training split. The epoch was chosen by
> video-level AUC on the validation split. Due to CPU constraints this was done for a single
> identity-grouped train/validation split, and the AVFF model was trained and its threshold
> selected on the same split before a single evaluation on the held-out test identities.

Speed tips for step 2 and 3 (measured on this laptop: 3.5 training img/s, 1.1 videos/s):
close Chrome and extra VS Code windows first, because with < 1 GB free RAM Windows starts paging
and everything slows down. Step 2 needs about 30–45 min to build the frame bank before training.
If the ETA printed in step 2 is too long, stop it (Ctrl+C), and rerun with fewer steps, e.g.
`python finetune_backbone.py --steps 150` (delete `backbone_ft\last_fold1.pth` first so it
starts fresh with the new schedule).
