#!/usr/bin/env python
# coding: utf-8
"""
Multimodal Deepfake Detection on the FULL FakeAVCeleb v1.2 (CPU-only)  --  v3
=============================================================================
Two-stage AVFF-style pipeline (Oorloff et al., CVPR 2024), adapted for a
CPU-only laptop and the severe 1:42 real/fake imbalance.

What changed vs. v2 (and why)
-----------------------------
DATA / EVALUATION (leakage fixes - required for a publishable protocol)
  * ALL videos are used (no 1000-sample subsampling).
  * Identity-grouped splits (StratifiedGroupKFold): one person never appears in
    both train and test, or both train and validation.
  * Stage-1 self-supervised pre-training uses ONLY the real videos of the
    current fold's TRAIN split (v2 used every real video, incl. test ones).
  * Threshold chosen on VALIDATION data only, then frozen and applied once to
    the held-out test set.  No model/fold is ever selected using test scores
    (v2 picked the "best fold" by test F1).

VIDEO FEATURES (ResNet-18 only, as requested)
  * Proper ImageNet ResNet-18 (strict weight loading - no silent random init),
    RGB input + ImageNet normalisation (v2 fed un-normalised BGR frames).
  * 224x224 frames (optional face crop), 16 frames per video.
  * Per frame: layer4 global-avg + global-max pool + layer3 avg pool
    (1280-d).  Max-pooling keeps localised blending artefacts that average
    pooling washes out.  Raw features are cached (v2 cached the output of a
    RANDOM 512->256 projection, which destroyed most video signal: AUC ~0.5).
  * Temporal-difference tokens (f_t - f_{t-1}) expose frame-to-frame flicker.

AUDIO FEATURES (same MFCC family as v2, richer)
  * 40 MFCC: per-slice mean + std + |delta| mean, + log-RMS energy (121-d),
    one slice per sampled video frame so audio/video tokens are aligned.

IMBALANCE HANDLING
  * AVFF Stage 1: self-supervised complementary masking + A2V/V2A cross-modal
    translation + audio-visual contrastive alignment, trained on REAL videos
    only, so the representation of "normal" talking faces is learned without
    any class bias.  Translation targets are the fixed normalised input
    features (cannot collapse).
  * Category-aware balanced sampling: 50% real, and the fake half split across
    RVFA / FVRA / FVFA so the rare RealVideo-FakeAudio type is not ignored.
  * Temporal-window augmentation: each 16-frame video yields 9 distinct
    8-frame windows -> many more distinct real samples.  Test-time: average
    over 3 windows.
  * Multi-task heads: besides real/fake, auxiliary "video manipulated?" and
    "audio manipulated?" heads (labels come for free from the 4 categories).
  * Label smoothing, AdamW + warm-up/cosine LR, early stopping on validation.

THRESHOLD
  * Searched over the full range of validation probabilities (v2 only scanned
    0.05-0.95 in 0.01 steps).  Chooses the threshold maximising macro F1 among
    thresholds where accuracy, macro precision, macro recall and macro F1 are
    all >= --target (default 0.80) on validation.  If no threshold satisfies
    it, the best worst-case threshold is used and a warning is printed.

Usage (Windows, from the folder that contains this file):
  python avff_v3_full.py --mode extract      # one-time, ~4-7 h, resumable
  python avff_v3_full.py --mode train        # ~30-60 min
  python avff_v3_full.py --mode all          # both

Fix 2 (fine-tuned backbone, single split; see RUN_GUIDE.md section 6):
  python finetune_backbone.py
  python avff_v3_full.py --mode extract --backbone_ckpt backbone_ft/resnet18_ft_fold1.pth \
      --audio_from features_v3 --cache_dir features_ft
  python avff_v3_full.py --mode train --folds 1 --backbone_ckpt backbone_ft/resnet18_ft_fold1.pth \
      --cache_dir features_ft --out_dir results_ft
"""

import os
import sys
import json
import time
import math
import argparse
import warnings
import subprocess
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

import eval_utils as E

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))


# =============================================================================
# PATHS
# =============================================================================

def first_existing(cands):
    for p in cands:
        if p and os.path.exists(p):
            return p
    return cands[0]


DEFAULT_DATASET_PATH = first_existing([
    os.environ.get("FAKEAVCELEB_PATH"),
    os.path.join(HERE, "FakeAVCeleb_v1.2"),
    os.path.join(HERE, "..", "FakeAVCeleb_v1.2"),
])
# ImageNet ResNet-18 weights (torchvision file name).  NOTE: do NOT point this
# at an older fine-tuned "best_model.pth" - it may have seen test identities.
DEFAULT_WEIGHTS_PATH = first_existing([
    os.environ.get("RESNET18_WEIGHTS"),
    os.path.join(HERE, "resnet18-f37072fd.pth"),
    os.path.join(HERE, "..", "resnet18-f37072fd.pth"),
])


def get_ffmpeg():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


FFMPEG_BINARY = get_ffmpeg()
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
SR = 16000


# =============================================================================
# DATASET DISCOVERY
# =============================================================================

def discover_dataset(root):
    samples = []
    print(f"Scanning dataset at: {root}")
    for cat, lab in E.CATEGORY_TO_LABEL.items():
        folder = os.path.join(root, cat)
        if not os.path.isdir(folder):
            print(f"  [Warning] missing folder: {folder}")
            continue
        for r, _, files in os.walk(folder):
            for f in sorted(files):
                if f.lower().endswith(".mp4"):
                    p = os.path.join(r, f)
                    samples.append({"path": p, "label": lab, "category": cat,
                                    "group": E.parse_identity(p)})
    samples.sort(key=lambda s: s["path"])
    return samples


# =============================================================================
# FEATURE EXTRACTION  (runs once, resumable)
# =============================================================================

import threading
_TLS = threading.local()


def face_detector():
    # one detector per thread (cv2 cascades are not guaranteed thread-safe)
    if getattr(_TLS, "face", None) is None:
        _TLS.face = cv2.CascadeClassifier(os.path.join(cv2.data.haarcascades,
                                                       "haarcascade_frontalface_default.xml"))
    return _TLS.face


def face_box(frame_rgb):
    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    faces = face_detector().detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,
                                             minSize=(40, 40))
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
    cx, cy, s = x + w / 2, y + h / 2, max(w, h) * 1.4      # 40% context margin
    H, W = frame_rgb.shape[:2]
    x0, y0 = int(max(0, cx - s / 2)), int(max(0, cy - s / 2))
    x1, y1 = int(min(W, cx + s / 2)), int(min(H, cy + s / 2))
    return x0, y0, x1, y1


def read_frames(path, n_frames, size, face_crop):
    """Uniformly samples n_frames RGB frames. Returns uint8 (n, size, size, 3) or None."""
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    if total > 0:
        want = np.linspace(0, max(total - 1, 0), n_frames).round().astype(int)
        want_set = {}
        for k, w in enumerate(want):
            want_set.setdefault(int(w), []).append(k)
        got = [None] * n_frames
        i = 0
        last = int(want[-1])
        while i <= last:
            if not cap.grab():                          # decode only, cheap
                break
            if i in want_set:
                ok, fr = cap.retrieve()
                if ok:
                    for k in want_set[i]:
                        got[k] = fr
            i += 1
        frames = [g for g in got if g is not None]
    cap.release()
    if not frames:
        return None
    while len(frames) < n_frames:                       # frame count over-reported
        frames.append(frames[-1])

    frames = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]
    H, W = frames[0].shape[:2]
    do_crop = face_crop == "on" or (face_crop == "auto" and min(H, W) >= 320)
    box = None
    if do_crop:
        try:
            box = face_box(frames[len(frames) // 2]) or face_box(frames[0])
        except Exception:
            box = None
        if box is None:                                 # no face found -> centre square
            s = min(H, W)
            box = ((W - s) // 2, (H - s) // 2, (W - s) // 2 + s, (H - s) // 2 + s)
    out = []
    for f in frames:
        if box is not None:
            x0, y0, x1, y1 = box
            f = f[y0:y1, x0:x1]
        out.append(cv2.resize(f, (size, size), interpolation=cv2.INTER_AREA))
    return np.stack(out).astype(np.uint8)


def load_audio(path):
    wav = os.path.splitext(path)[0] + ".wav"
    if os.path.exists(wav):
        import scipy.io.wavfile
        rate, raw = scipy.io.wavfile.read(wav)
        if raw.ndim > 1:
            raw = raw.mean(axis=1)
        if raw.dtype == np.int16:
            y = raw.astype(np.float32) / 32768.0
        elif raw.dtype == np.int32:
            y = raw.astype(np.float32) / 2147483648.0
        else:
            y = raw.astype(np.float32)
        if rate != SR:
            import torchaudio.functional as AF
            y = AF.resample(torch.from_numpy(y), rate, SR).numpy()
        return y
    cmd = [FFMPEG_BINARY, "-i", path, "-vn", "-ac", "1", "-ar", str(SR),
           "-f", "s16le", "-loglevel", "error", "-"]
    kw = {}
    if os.name == "nt":
        kw["creationflags"] = 0x08000000                # CREATE_NO_WINDOW
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, **kw)
    return np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32) / 32768.0


_MFCC = None


def audio_features(y, n_slices):
    """Returns (n_slices, 121) float32 and ok-flag.  Slices are aligned with the
    uniformly sampled video frames."""
    global _MFCC
    import torchaudio.transforms as T
    import torchaudio.functional as AF
    D = 121
    if y is None or len(y) < SR // 5:
        return np.zeros((n_slices, D), np.float32), False
    if _MFCC is None:
        _MFCC = T.MFCC(sample_rate=SR, n_mfcc=40,
                       melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 64})
    wav = torch.from_numpy(np.ascontiguousarray(y)).float().unsqueeze(0)
    with torch.no_grad():
        m = _MFCC(wav).squeeze(0)                       # (40, Ta)
        d = AF.compute_deltas(m.unsqueeze(0)).squeeze(0)
    Ta = m.shape[1]
    hop = 160
    cs = np.concatenate([[0.0], np.cumsum(y.astype(np.float64) ** 2)])
    centers = np.arange(Ta) * hop
    lo = np.clip(centers - 200, 0, len(y))
    hi = np.clip(centers + 200, 0, len(y))
    frame_rms = torch.from_numpy(
        np.sqrt((cs[hi] - cs[lo]) / np.maximum(hi - lo, 1) + 1e-8)).float()
    if frame_rms.shape[0] < Ta:
        frame_rms = F.pad(frame_rms, (0, Ta - frame_rms.shape[0]), value=float(frame_rms[-1]))
    edges = np.linspace(0, Ta, n_slices + 1).round().astype(int)
    out = np.zeros((n_slices, D), np.float32)
    for i in range(n_slices):
        a, b = edges[i], max(edges[i + 1], edges[i] + 1)
        b = min(b, Ta)
        a = min(a, b - 1)
        seg, dseg = m[:, a:b], d[:, a:b]
        out[i, :40] = seg.mean(1).numpy()
        out[i, 40:80] = seg.std(1, unbiased=False).numpy()
        out[i, 80:120] = dseg.abs().mean(1).numpy()
        out[i, 120] = float(torch.log(frame_rms[a:b].mean() + 1e-6))
    return out, True


def load_one(sample, n_frames, size, face_crop, audio=None):
    """audio: optional precomputed (features, ok) -> skips audio decoding."""
    try:
        fr = read_frames(sample["path"], n_frames, size, face_crop)
    except Exception:
        fr = None
    if audio is not None:
        return fr, audio[0], audio[1]
    try:
        af, a_ok = audio_features(load_audio(sample["path"]), n_frames)
    except Exception:
        af, a_ok = np.zeros((n_frames, 121), np.float32), False
    return fr, af, a_ok


def file_sha1(path):
    import hashlib
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()[:16]


def load_backbone_ckpt(path):
    """Fine-tuned backbone written by finetune_backbone.py."""
    ck = torch.load(path, map_location="cpu")
    if not (isinstance(ck, dict) and {"backbone", "head", "meta"} <= set(ck)):
        raise RuntimeError(f"{path} is not a checkpoint written by finetune_backbone.py")
    return ck


def check_backbone_leakage(meta, groups, eval_sets, log):
    """The fine-tuned backbone must never have seen a validation or test identity
    of the folds being run.  eval_sets: list of (name, index array)."""
    seen = set(meta["train_groups"])
    for name, idx in eval_sets:
        bad = seen & set(groups[idx])
        if bad:
            sys.exit(f"[Error] Leakage: the fine-tuned backbone was trained on {len(bad)} "
                     f"identities that are in {name} (e.g. {sorted(map(str, bad))[:3]}). It was trained for "
                     f"fold {meta['fold']} (seed={meta['seed']}, n_folds={meta['n_folds']}, "
                     f"test_frac={meta['test_frac']}); run only that fold with the same settings.")
    log(f"  [Backbone] leakage check passed: none of its {len(seen)} training identities "
        f"appear in validation or test.")


class ResNet18Features(nn.Module):
    """Frozen ResNet-18 -> per-frame [l4 avg | l4 max | l3 avg] = 1280-d.
    With a fine-tuned checkpoint (Fix 2), the frame-level fake logit margin is
    appended -> 1281-d."""

    def __init__(self, weights_path, backbone_ckpt=None):
        super().__init__()
        from torchvision.models import resnet18
        net = resnet18(weights=None)
        self.head = None
        if backbone_ckpt:
            ck = load_backbone_ckpt(backbone_ckpt)
            net.load_state_dict(ck["backbone"], strict=True)
            self.head = nn.Linear(512, 2)
            self.head.load_state_dict(ck["head"], strict=True)
            self.head.eval()
            print(f"  [ResNet-18] loading FINE-TUNED backbone from {backbone_ckpt} "
                  f"(fold {ck['meta']['fold']}, best epoch {ck['meta'].get('epoch')})")
            self.net = net.eval()
            for p in self.parameters():
                p.requires_grad_(False)
            return
        sd = None
        if weights_path and os.path.exists(weights_path):
            sd = torch.load(weights_path, map_location="cpu")
            if isinstance(sd, dict) and "state_dict" in sd:
                sd = sd["state_dict"]
            print(f"  [ResNet-18] loading ImageNet weights from {weights_path}")
        else:
            try:
                from torchvision.models import ResNet18_Weights
                print("  [ResNet-18] local weights not found -> downloading ImageNet weights")
                sd = ResNet18_Weights.IMAGENET1K_V1.get_state_dict(progress=True)
            except Exception as e:
                raise RuntimeError(
                    f"ImageNet ResNet-18 weights not found at '{weights_path}' and download "
                    f"failed ({e}). Pass --weights_path to resnet18-f37072fd.pth") from e
        missing, unexpected = net.load_state_dict(sd, strict=False)
        missing = [k for k in missing if not k.startswith("fc.")]
        if missing:
            raise RuntimeError(f"ResNet-18 weights do not match ({len(missing)} missing keys, "
                               f"e.g. {missing[:3]}). Use the torchvision ImageNet file "
                               f"resnet18-f37072fd.pth.")
        self.net = net.eval()
        for p in self.net.parameters():
            p.requires_grad_(False)

    @torch.inference_mode()
    def forward(self, x):
        n = self.net
        x = n.maxpool(n.relu(n.bn1(n.conv1(x))))
        x = n.layer2(n.layer1(x))
        l3 = n.layer3(x)
        l4 = n.layer4(l3)
        l4avg = l4.mean((2, 3))
        out = [l4avg, l4.amax((2, 3)), l3.mean((2, 3))]
        if self.head is not None:                       # frame-level P(fake video) margin
            lg = self.head(l4avg)
            out.append((lg[:, 1] - lg[:, 0]).unsqueeze(1))
        return torch.cat(out, dim=1)


def shard_files(cache_dir):
    if not os.path.isdir(cache_dir):
        return []
    return sorted(os.path.join(cache_dir, f) for f in os.listdir(cache_dir)
                  if f.startswith("shard_") and f.endswith(".pt"))


def extract_features(samples, args):
    os.makedirs(args.cache_dir, exist_ok=True)
    meta_path = os.path.join(args.cache_dir, "extract_config.json")
    video_dim = 1281 if args.backbone_ckpt else 1280
    cfg = {"n_frames": args.n_frames, "img_size": args.img_size, "face_crop": args.face_crop,
           "video_dim": video_dim, "audio_dim": 121,
           "backbone": file_sha1(args.backbone_ckpt) if args.backbone_ckpt else "imagenet"}
    if os.path.exists(meta_path):
        old = json.load(open(meta_path))
        old.setdefault("backbone", "imagenet")          # caches written before Fix 2
        if any(old.get(k) != v for k, v in cfg.items()):
            sys.exit(f"[Error] {args.cache_dir} was built with {old}; you asked for {cfg}. "
                     f"Use a different --cache_dir.")
    json.dump(cfg, open(meta_path, "w"), indent=2)

    done = set()
    for sf in shard_files(args.cache_dir):
        done.update(torch.load(sf, map_location="cpu")["paths"])
    todo = [s for s in samples if s["path"] not in done]
    print(f"  [Extract] {len(done)} videos already cached, {len(todo)} to go.")
    if not todo:
        return

    audio_cache = {}
    if args.audio_from:                                 # reuse audio features (identical MFCCs)
        old_cfg = json.load(open(os.path.join(args.audio_from, "extract_config.json")))
        if old_cfg.get("n_frames") != args.n_frames:
            sys.exit(f"[Error] --audio_from {args.audio_from} has n_frames="
                     f"{old_cfg.get('n_frames')}, need {args.n_frames}")
        want = {s["path"] for s in todo}
        for sf in shard_files(args.audio_from):
            d = torch.load(sf, map_location="cpu")
            for i, p in enumerate(d["paths"]):
                if p in want:
                    audio_cache[p] = (d["a"][i].numpy(), bool(d["a_ok"][i]))
            del d
        print(f"  [Extract] audio reused from {args.audio_from} for "
              f"{len(audio_cache)}/{len(todo)} videos (rest decoded)")

    torch.set_num_threads(args.threads)
    model = ResNet18Features(args.weights_path, args.backbone_ckpt)
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    chunk = 32
    chunks = [todo[i:i + chunk] for i in range(0, len(todo), chunk)]
    shard_id = len(shard_files(args.cache_dir))
    buf = {"paths": [], "v": [], "a": [], "v_ok": [], "a_ok": []}

    def flush():
        nonlocal shard_id, buf
        if not buf["paths"]:
            return
        out = os.path.join(args.cache_dir, f"shard_{shard_id:05d}.pt")
        torch.save({"paths": buf["paths"], "v": torch.stack(buf["v"]),
                    "a": torch.stack(buf["a"]), "v_ok": torch.tensor(buf["v_ok"]),
                    "a_ok": torch.tensor(buf["a_ok"])}, out + ".tmp")
        os.replace(out + ".tmp", out)                   # atomic: crash-safe
        shard_id += 1
        buf = {"paths": [], "v": [], "a": [], "v_ok": [], "a_ok": []}

    t0 = time.time()
    n_done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        submit = lambda c: [ex.submit(load_one, s, args.n_frames, args.img_size, args.face_crop,
                                      audio_cache.get(s["path"]))
                            for s in c]
        pending = submit(chunks[0])
        for ci, c in enumerate(chunks):
            results = [f.result() for f in pending]
            if ci + 1 < len(chunks):
                pending = submit(chunks[ci + 1])       # decode next chunk while CNN runs
            imgs, owners = [], []
            for j, (fr, _, _) in enumerate(results):
                if fr is not None:
                    imgs.append(fr)
                    owners.append(j)
            feats = {}
            if imgs:
                x = torch.from_numpy(np.concatenate(imgs)).permute(0, 3, 1, 2).float().div_(255)
                x = ((x - mean) / std).contiguous(memory_format=torch.channels_last)
                outs = [model(x[i:i + 128]) for i in range(0, x.shape[0], 128)]
                fv = torch.cat(outs).view(len(imgs), args.n_frames, -1)
                for j, k in enumerate(owners):
                    feats[k] = fv[j]
            for j, s in enumerate(c):
                fr, af, a_ok = results[j]
                buf["paths"].append(s["path"])
                buf["v"].append((feats[j] if j in feats else
                                 torch.zeros(args.n_frames, video_dim)).half())
                buf["a"].append(torch.from_numpy(af).float())
                buf["v_ok"].append(j in feats)
                buf["a_ok"].append(bool(a_ok))
            n_done += len(c)
            if len(buf["paths"]) >= 512:
                flush()
            el = time.time() - t0
            rate = n_done / max(el, 1e-6)
            eta = (len(todo) - n_done) / max(rate, 1e-6)
            print(f"\r  [Extract] {n_done}/{len(todo)} videos | {rate:.2f} vid/s | "
                  f"elapsed {el/3600:.2f} h | ETA {eta/3600:.2f} h   ", end="", flush=True)
    flush()
    print(f"\n  [Extract] done in {(time.time()-t0)/3600:.2f} h")


def load_cache(samples, args):
    by_path = {}
    for sf in shard_files(args.cache_dir):
        d = torch.load(sf, map_location="cpu")
        for i, p in enumerate(d["paths"]):
            by_path[p] = (d["v"][i], d["a"][i], bool(d["v_ok"][i]), bool(d["a_ok"][i]))
    keep, V, A = [], [], []
    n_missing = n_bad = n_noaudio = 0
    for s in samples:
        r = by_path.get(s["path"])
        if r is None:
            n_missing += 1
            continue
        if not r[2]:
            n_bad += 1
            continue
        n_noaudio += (not r[3])
        keep.append(s)
        V.append(r[0])
        A.append(r[1])
    print(f"  [Cache] usable={len(keep)} | not yet extracted={n_missing} | "
          f"undecodable video (excluded)={n_bad} | no audio (zeros)={n_noaudio}")
    if n_missing:
        print("  [Warning] Some videos are not cached; run --mode extract to finish.")
    return keep, torch.stack(V), torch.stack(A)       # V fp16 (N,F,1280), A fp32 (N,F,121)


# =============================================================================
# MODEL  (AVFF-style)
# =============================================================================

class TemporalEncoder(nn.Module):
    def __init__(self, d_in, d, T, layers=1, heads=4, drop=0.1):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(d_in, d), nn.LayerNorm(d), nn.GELU(), nn.Dropout(drop))
        self.pos = nn.Parameter(torch.zeros(1, T, d))
        nn.init.normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(d, heads, 2 * d, dropout=drop, batch_first=True,
                                           norm_first=True)
        self.tr = nn.TransformerEncoder(layer, layers)

    def forward(self, x):
        return self.tr(self.inp(x) + self.pos[:, :x.shape[1]])


class CrossModalTranslator(nn.Module):
    """A2V / V2A: predicts the (normalised) features of the other modality."""

    def __init__(self, d_in, d_out, T, hidden=256, layers=2, heads=4):
        super().__init__()
        self.enc = TemporalEncoder(d_in, hidden, T, layers, heads)
        self.out = nn.Linear(hidden, d_out)

    def forward(self, x):
        return self.out(self.enc(x))


class AVFFv3(nn.Module):
    def __init__(self, dv_raw, da_raw, T, d_v=256, d_a=128, d_clip=128):
        super().__init__()
        self.T = T
        self.register_buffer("v_mean", torch.zeros(dv_raw))
        self.register_buffer("v_std", torch.ones(dv_raw))
        self.register_buffer("a_mean", torch.zeros(da_raw))
        self.register_buffer("a_std", torch.ones(da_raw))
        # video: [frame feature | temporal difference]
        self.video_encoder = TemporalEncoder(2 * dv_raw, d_v, T, layers=1, drop=0.2)
        self.audio_encoder = TemporalEncoder(da_raw, d_a, T, layers=1, drop=0.1)
        self.a2v = CrossModalTranslator(d_a, dv_raw, T)
        self.v2a = CrossModalTranslator(d_v, da_raw, T)
        self.v_clip = nn.Linear(d_v, d_clip)
        self.a_clip = nn.Linear(d_a, d_clip)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))
        self.cross_attn = nn.MultiheadAttention(d_v, 4, batch_first=True)
        self.audio_to_v = nn.Linear(d_a, d_v)
        disc_dim = 4 * T + 3
        head_in = d_v + d_v + d_a + disc_dim
        self.classifier = nn.Sequential(
            nn.LayerNorm(head_in), nn.Linear(head_in, 256), nn.GELU(), nn.Dropout(0.5),
            nn.Linear(256, 64), nn.GELU(), nn.Dropout(0.3), nn.Linear(64, 2))
        self.video_head = nn.Sequential(nn.Dropout(0.3), nn.Linear(d_v, 1))
        self.audio_head = nn.Sequential(nn.Dropout(0.3), nn.Linear(d_a, 1))

    def set_norm(self, vm, vs, am, as_):
        self.v_mean.copy_(vm); self.v_std.copy_(vs)
        self.a_mean.copy_(am); self.a_std.copy_(as_)

    def normalise(self, v, a, noise=0.0):
        v = (v.float() - self.v_mean) / self.v_std
        a = (a.float() - self.a_mean) / self.a_std
        if noise > 0:
            v = v + noise * torch.randn_like(v)
            a = a + noise * torch.randn_like(a)
        return v, a

    def encode(self, vn, an):
        dv = torch.cat([torch.zeros_like(vn[:, :1]), vn[:, 1:] - vn[:, :-1]], dim=1)
        return self.video_encoder(torch.cat([vn, dv], -1)), self.audio_encoder(an)

    # ---------------- Stage 1: self-supervised, REAL videos only -------------
    def forward_stage1(self, v, a, noise=0.0):
        vn, an = self.normalise(v, a, noise)
        vt, at = self.encode(vn, an)
        B, T, _ = vt.shape
        # complementary masks: video visible where M_v=1, audio visible where M_v=0
        r = torch.rand(B, T, device=vt.device).argsort(1)
        M_v = (r < T // 2).float()
        M_a = 1.0 - M_v
        pred_v = self.a2v(at * M_a.unsqueeze(-1))       # audio visible -> predict video
        pred_a = self.v2a(vt * M_v.unsqueeze(-1))       # video visible -> predict audio
        mv, ma = (1 - M_v).unsqueeze(-1), (1 - M_a).unsqueeze(-1)
        loss_v = ((pred_v - vn) ** 2 * mv).sum() / (mv.sum() * vn.shape[-1])
        loss_a = ((pred_a - an) ** 2 * ma).sum() / (ma.sum() * an.shape[-1])
        zv = F.normalize(self.v_clip(vt.mean(1)), dim=-1)
        za = F.normalize(self.a_clip(at.mean(1)), dim=-1)
        logits = self.logit_scale.exp().clamp(max=100) * zv @ za.t()
        tgt = torch.arange(B, device=vt.device)
        loss_c = (F.cross_entropy(logits, tgt) + F.cross_entropy(logits.t(), tgt)) / 2
        loss = loss_v + loss_a + 0.5 * loss_c
        return loss, {"A2V": loss_v.item(), "V2A": loss_a.item(), "NCE": loss_c.item()}

    # ---------------- Stage 2: classification --------------------------------
    def forward_stage2(self, v, a, noise=0.0):
        vn, an = self.normalise(v, a, noise)
        vt, at = self.encode(vn, an)
        pred_v, pred_a = self.a2v(at), self.v2a(vt)
        e_v = ((vn - pred_v).pow(2).mean(-1) + 1e-8).sqrt()
        e_a = ((an - pred_a).pow(2).mean(-1) + 1e-8).sqrt()
        c_v = F.cosine_similarity(vn, pred_v, dim=-1)
        c_a = F.cosine_similarity(an, pred_a, dim=-1)
        zv = F.normalize(self.v_clip(vt.mean(1)), dim=-1)
        za = F.normalize(self.a_clip(at.mean(1)), dim=-1)
        sync = (zv * za).sum(-1, keepdim=True)
        disc = torch.cat([e_v, e_a, c_v, c_a, e_v.mean(-1, keepdim=True),
                          e_a.mean(-1, keepdim=True), sync], -1)
        a_as_v = self.audio_to_v(at)
        fused, _ = self.cross_attn(vt, a_as_v, a_as_v)
        v_pool, a_pool = vt.mean(1), at.mean(1)
        h = torch.cat([fused.mean(1), v_pool, a_pool, disc], -1)
        return self.classifier(h), self.video_head(v_pool).squeeze(-1), \
            self.audio_head(a_pool).squeeze(-1)


# =============================================================================
# TRAINING UTILITIES
# =============================================================================

def take_windows(X, idx, T, starts):
    """X: (N,F,D) -> (B,T,D) windows starting at `starts` for rows `idx`."""
    t = starts.view(-1, 1) + torch.arange(T).view(1, -1)
    return X[idx.view(-1, 1), t]


def sampler_weights(labels4, shares):
    labels4 = np.asarray(labels4)
    counts = np.bincount(labels4, minlength=4).astype(float)
    sh = np.array([shares[c] if counts[c] > 0 else 0.0 for c in range(4)])
    sh = sh / sh.sum()
    return torch.tensor(sh[labels4] / counts[labels4], dtype=torch.double)


def lr_lambda(total, warm):
    return lambda s: (s + 1) / warm if s < warm else \
        0.5 * (1 + math.cos(math.pi * min(1.0, (s - warm) / max(1, total - warm))))


@torch.no_grad()
def predict(model, V, A, idx, T, bs=256):
    """P(fake) averaged over ALL temporal windows of length T (dense sliding, as in AVFF)."""
    model.eval()
    Fn = V.shape[1]
    starts_list = list(range(0, Fn - T + 1))
    idx = torch.as_tensor(idx)
    probs = torch.zeros(len(idx))
    for s0 in starts_list:
        for i in range(0, len(idx), bs):
            b = idx[i:i + bs]
            st = torch.full((len(b),), s0, dtype=torch.long)
            logits, _, _ = model.forward_stage2(take_windows(V, b, T, st), take_windows(A, b, T, st))
            probs[i:i + bs] += torch.softmax(logits, -1)[:, 1]
    return (probs / len(starts_list)).numpy()


def _mean_std(X, idx, chunk=1024):
    s = torch.zeros(X.shape[-1], dtype=torch.float64)
    s2 = torch.zeros_like(s)
    n = 0
    for i in range(0, len(idx), chunk):
        x = X[torch.as_tensor(idx[i:i + chunk])].double().reshape(-1, X.shape[-1])
        s += x.sum(0); s2 += (x * x).sum(0); n += x.shape[0]
    m = s / n
    sd = (s2 / n - m * m).clamp_min(0).sqrt()
    return m.float(), sd.float().clamp_min(1e-4)


def fit_norm(model, V, A, idx):
    """Per-dimension normalisation fitted on the TRAIN split only (chunked: low RAM)."""
    vm, vs = _mean_std(V, idx)
    am, as_ = _mean_std(A, idx)
    model.set_norm(vm, vs, am, as_)


def train_stage1(model, V, A, real_idx, args, log):
    T, Fn = args.T, V.shape[1]
    real_idx = torch.as_tensor(real_idx)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr_s1, weight_decay=0.05)
    steps_per_epoch = max(1, math.ceil(len(real_idx) * args.s1_views / args.batch_size))
    total = steps_per_epoch * args.epochs_s1
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(total, max(1, total // 20)))
    model.train()
    for ep in range(args.epochs_s1):
        agg = {"loss": 0, "A2V": 0, "V2A": 0, "NCE": 0}
        for _ in range(steps_per_epoch):
            b = real_idx[torch.randint(len(real_idx), (args.batch_size,))]
            st = torch.randint(0, Fn - T + 1, (len(b),))
            loss, m = model.forward_stage1(take_windows(V, b, T, st), take_windows(A, b, T, st),
                                           noise=args.noise)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sch.step()
            agg["loss"] += loss.item()
            for k in m:
                agg[k] += m[k]
        log(f"    Stage1 ep {ep+1:02d}/{args.epochs_s1} | " +
            " ".join(f"{k}={v/steps_per_epoch:.4f}" for k, v in agg.items()))


def train_stage2(model, V, A, y4, tr_idx, va_idx, args, ckpt_path, log):
    T, Fn = args.T, V.shape[1]
    y4 = torch.as_tensor(y4)
    ybin = (y4 > 0).long()
    yvid = (y4 >= 2).float()                     # video manipulated
    yaud = ((y4 == 1) | (y4 == 3)).float()       # audio manipulated
    tr = torch.as_tensor(tr_idx)
    w = sampler_weights(y4[tr].numpy(), args.shares)
    steps = max(1, math.ceil(len(tr) * args.epoch_frac / args.batch_size))
    total = steps * args.epochs_s2
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(total, max(1, total // 20)))
    ce = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    import copy
    ema = copy.deepcopy(model)                   # exponential moving average of weights:
    for p_ in ema.parameters():                  # much less noisy than the raw weights
        p_.requires_grad_(False)
    bce = nn.BCEWithLogitsLoss()
    best, bad = -1.0, 0
    yva = ybin[torch.as_tensor(va_idx)].numpy()
    for ep in range(args.epochs_s2):
        model.train()
        tl = 0.0
        for _ in range(steps):
            b = tr[torch.multinomial(w, args.batch_size, replacement=True)]
            st = torch.randint(0, Fn - T + 1, (len(b),))
            logits, lv, la = model.forward_stage2(take_windows(V, b, T, st),
                                                  take_windows(A, b, T, st), noise=args.noise)
            loss = ce(logits, ybin[b]) + args.aux_weight * (bce(lv, yvid[b]) + bce(la, yaud[b]))
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sch.step()
            with torch.no_grad():
                for pe, pm in zip(ema.parameters(), model.parameters()):
                    pe.mul_(args.ema).add_(pm.detach(), alpha=1 - args.ema)
            tl += loss.item()
        pva = predict(ema, V, A, va_idx, T)
        thr, info = E.select_threshold(yva, pva, args.target, args.thr_objective, args.thr_view)
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(yva, pva)
        # model selection on VALIDATION (AUC is threshold-free and less noisy with 80 reals)
        score = (1.0 if info["feasible"] else 0.0) + 0.5 * info["macro_f1"] + 0.5 * auc
        flag = "OK " if info["feasible"] else "-- "
        log(f"    Stage2 ep {ep+1:02d}/{args.epochs_s2} | loss={tl/steps:.4f} | val AUC={auc:.4f} | "
            f"{flag}thr={thr:.4f} Acc={info['accuracy']:.3f} P={info['macro_precision']:.3f} "
            f"R={info['macro_recall']:.3f} F1={info['macro_f1']:.3f}")
        if score > best:
            best, bad = score, 0
            torch.save(ema.state_dict(), ckpt_path)
        else:
            bad += 1
            if bad >= args.patience:
                log(f"    early stop (no val improvement for {args.patience} epochs)")
                break
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    return model


# =============================================================================
# MAIN
# =============================================================================

def main():
    ap = argparse.ArgumentParser(description="AVFF-style multimodal deepfake detection v3 (CPU)")
    ap.add_argument("--mode", default="all", choices=["extract", "train", "all"])
    ap.add_argument("--dataset_path", default=DEFAULT_DATASET_PATH)
    ap.add_argument("--weights_path", default=DEFAULT_WEIGHTS_PATH,
                    help="ImageNet ResNet-18 weights (resnet18-f37072fd.pth)")
    ap.add_argument("--cache_dir", default=os.path.join(HERE, "features_v3"))
    ap.add_argument("--out_dir", default=os.path.join(HERE, "results_v3"))
    ap.add_argument("--max_samples", type=int, default=0, help="0 = use ALL videos")
    # extraction
    ap.add_argument("--n_frames", type=int, default=16)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--face_crop", default="auto", choices=["auto", "on", "off"])
    ap.add_argument("--workers", type=int, default=3, help="decode threads")
    ap.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 4)))
    ap.add_argument("--backbone_ckpt", default="",
                    help="fine-tuned ResNet-18 from finetune_backbone.py (Fix 2); needed for "
                         "both extract and train when the cache was built with it")
    ap.add_argument("--audio_from", default="",
                    help="existing cache dir to copy audio features from (skips audio decoding)")
    # training
    ap.add_argument("--folds", default="all",
                    help="which folds to run, e.g. '1' or '1,3' (default: all)")
    ap.add_argument("--T", type=int, default=8, help="frames per training window")
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--test_frac", type=float, default=0.2)
    ap.add_argument("--epochs_s1", type=int, default=30)
    ap.add_argument("--s1_views", type=int, default=4, help="windows per real video per S1 epoch")
    ap.add_argument("--epochs_s2", type=int, default=40)
    ap.add_argument("--epoch_frac", type=float, default=0.5,
                    help="fraction of the train set drawn per Stage-2 epoch")
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--ema", type=float, default=0.99, help="EMA decay of weights (per step)")
    ap.add_argument("--lr_s1", type=float, default=5e-4)
    ap.add_argument("--noise", type=float, default=0.10, help="feature-noise augmentation")
    ap.add_argument("--label_smoothing", type=float, default=0.05)
    ap.add_argument("--aux_weight", type=float, default=0.5)
    ap.add_argument("--shares", type=float, nargs=4, default=[0.5, 0.1, 0.2, 0.2],
                    help="sampling share for RVRA RVFA FVRA FVFA")
    ap.add_argument("--target", type=float, default=0.80)
    ap.add_argument("--thr_objective", default="macro_f1", choices=["macro_f1", "min_metric"])
    ap.add_argument("--thr_view", default="natural", choices=["natural", "class_balanced"],
                    help="distribution the threshold is optimised for (natural = real test ratio)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(args.threads)
    os.makedirs(args.out_dir, exist_ok=True)
    log_f = open(os.path.join(args.out_dir, "train_log.txt"), "a", encoding="utf-8")

    def log(msg=""):
        print(msg)
        log_f.write(msg + "\n")
        log_f.flush()

    log("=" * 78)
    log(f"AVFF v3 | {time.strftime('%Y-%m-%d %H:%M:%S')} | CPU threads={args.threads}")
    log(f"Dataset: {args.dataset_path}")
    log("Args: " + json.dumps(vars(args)))

    samples = discover_dataset(args.dataset_path)
    if not samples:
        sys.exit(f"[Error] No videos found under {args.dataset_path}")
    if args.max_samples and args.max_samples < len(samples):   # debugging only
        rng = np.random.default_rng(args.seed)
        samples = [samples[i] for i in sorted(rng.choice(len(samples), args.max_samples, False))]
        log(f"[Debug] random subset of {len(samples)} videos (NOT for reported results)")
    cnt = np.bincount([s["label"] for s in samples], minlength=4)
    log(f"Videos: {len(samples)} | RVRA={cnt[0]} RVFA={cnt[1]} FVRA={cnt[2]} FVFA={cnt[3]} | "
        f"real:fake = 1:{(len(samples)-cnt[0])/max(cnt[0],1):.1f} | "
        f"identities={len(set(s['group'] for s in samples))}")

    if args.mode in ("extract", "all"):
        extract_features(samples, args)
        if args.mode == "extract":
            return

    samples, V, A = load_cache(samples, args)
    if V.shape[1] < args.T:
        sys.exit(f"[Error] --T {args.T} > cached frames {V.shape[1]}")
    y4 = np.array([s["label"] for s in samples])
    ybin = (y4 > 0).astype(int)
    groups = np.array([s["group"] for s in samples])

    dev_idx, test_idx, folds = E.make_grouped_splits(y4, groups, args.test_frac, args.n_folds,
                                                     args.seed)
    log(f"Identity-grouped split: dev={len(dev_idx)} ({(ybin[dev_idx]==0).sum()} real) | "
        f"test={len(test_idx)} ({(ybin[test_idx]==0).sum()} real) | "
        f"test identities never seen in training/validation/threshold tuning.")
    json.dump({"test_paths": [samples[i]["path"] for i in test_idx]},
              open(os.path.join(args.out_dir, "test_split.json"), "w"))

    run_folds = (list(range(len(folds))) if args.folds == "all" else
                 sorted({int(x) - 1 for x in args.folds.split(",")}))
    if not run_folds or min(run_folds) < 0 or max(run_folds) >= len(folds):
        sys.exit(f"[Error] --folds {args.folds}: folds are numbered 1..{len(folds)}")

    # Fix 2 leakage guard: a cache built with a fine-tuned backbone may only be used
    # for folds whose validation and test identities that backbone never saw.
    cache_cfg = json.load(open(os.path.join(args.cache_dir, "extract_config.json")))
    cache_bb = cache_cfg.get("backbone", "imagenet")
    if cache_bb != "imagenet":
        if not args.backbone_ckpt:
            sys.exit("[Error] This cache was extracted with a fine-tuned backbone; pass the same "
                     "--backbone_ckpt so the leakage check can run.")
        if file_sha1(args.backbone_ckpt) != cache_bb:
            sys.exit(f"[Error] --backbone_ckpt does not match the backbone that built "
                     f"{args.cache_dir}.")
        meta = load_backbone_ckpt(args.backbone_ckpt)["meta"]
        check_backbone_leakage(meta, groups, [("test", test_idx)] +
                               [(f"fold {k+1} validation", folds[k][1]) for k in run_folds], log)
    elif args.backbone_ckpt:
        log("  [Warning] --backbone_ckpt ignored: this cache was built with the ImageNet backbone.")

    oof = np.full(len(samples), np.nan)
    test_probs, fold_rows = [], []
    for k, (tr, va) in enumerate(folds):
        if k not in run_folds:
            continue
        log("-" * 78)
        log(f"FOLD {k+1}/{args.n_folds} | train={len(tr)} ({(ybin[tr]==0).sum()} real) | "
            f"val={len(va)} ({(ybin[va]==0).sum()} real)")
        torch.manual_seed(args.seed + k)
        model = AVFFv3(V.shape[-1], A.shape[-1], args.T)
        fit_norm(model, V, A, tr)                       # statistics from TRAIN only
        real_tr = tr[y4[tr] == 0]                       # RVRA of this fold's TRAIN split
        log(f"  Stage 1 (self-supervised, {len(real_tr)} real train videos only)")
        train_stage1(model, V, A, real_tr, args, log)
        log("  Stage 2 (supervised)")
        ckpt = os.path.join(args.out_dir, f"avff_v3_fold{k+1}.pth")
        model = train_stage2(model, V, A, y4, tr, va, args, ckpt, log)

        pva = predict(model, V, A, va, args.T)
        oof[va] = pva
        thr_k, info_k = E.select_threshold(ybin[va], pva, args.target, args.thr_objective,
                                           args.thr_view)
        pte = predict(model, V, A, test_idx, args.T)    # threshold already frozen
        test_probs.append(pte)
        rep_k = E.full_report(ybin[test_idx], pte, thr_k)
        log(E.format_report(rep_k, f"  Fold {k+1} TEST (threshold {thr_k:.4f} fixed from its "
                                   f"validation fold, val-feasible={info_k['feasible']})"))
        fold_rows.append({"fold": k + 1, "val_threshold_info": info_k, "test": rep_k})

    # ---------------- Final: ensemble of the fold models -----------------------
    log("=" * 78)
    oof_idx = dev_idx[~np.isnan(oof[dev_idx])]          # only folds that were run
    dev_y, dev_p = ybin[oof_idx], oof[oof_idx]
    thr, info = E.select_threshold(dev_y, dev_p, args.target, args.thr_objective, args.thr_view)
    log(f"Threshold selected on out-of-fold VALIDATION predictions ({len(oof_idx)} videos): "
        f"{thr:.4f} | feasible(all>= {args.target:.2f})={info['feasible']} | "
        f"val Acc={info['accuracy']:.4f} P={info['macro_precision']:.4f} "
        f"R={info['macro_recall']:.4f} F1={info['macro_f1']:.4f}")
    if not info["feasible"]:
        log("  [Warning] No threshold reaches the target on validation; reporting the "
            "threshold with the best worst-case metric. Do NOT re-tune on test.")
    log(E.format_report(E.full_report(dev_y, dev_p, thr), "OOF VALIDATION (dev set)"))

    # Ensemble: each fold's probability is converted to a margin relative to the
    # threshold chosen on THAT fold's validation split: z_k = logit(p_k) - logit(thr_k).
    # Averaging margins keeps every fold's validated operating point (averaging raw
    # probabilities and re-using one OOF threshold shifts the operating point).
    def _logit(x):
        x = np.clip(np.asarray(x, dtype=float), 1e-6, 1 - 1e-6)
        return np.log(x / (1 - x))
    z = np.mean([_logit(p) - _logit(r["val_threshold_info"]["threshold"])
                 for p, r in zip(test_probs, fold_rows)], axis=0)
    p_test = 1.0 / (1.0 + np.exp(-z))                 # 0.5 == every fold's own threshold
    thr = 0.5
    rep = E.full_report(ybin[test_idx], p_test, thr)
    name = (f"{len(fold_rows)}-fold margin ensemble" if len(fold_rows) > 1 else
            f"single model, fold {fold_rows[0]['fold']}")
    log(E.format_report(rep, f"*** HELD-OUT TEST ({name}; each fold's threshold "
                             "fixed from its validation split; evaluated once) ***"))

    keys = ["accuracy", "macro_precision", "macro_recall", "macro_f1"]
    arr = np.array([[r["test"]["natural"][kk] for kk in keys] + [r["test"]["auc"]]
                    for r in fold_rows])
    log("Per-fold TEST (each with its own validation threshold), mean +- std:")
    for j, kk in enumerate(keys + ["auc"]):
        log(f"  {kk:>16}: {arr[:, j].mean():.4f} +- {arr[:, j].std():.4f}")

    # per-category recall on test (which manipulation types are missed?)
    pred = (p_test >= thr).astype(int)
    for c, name in enumerate(E.CATEGORY_TO_LABEL):
        m = y4[test_idx] == c
        if m.any():
            acc_c = (pred[m] == (1 if c > 0 else 0)).mean()
            log(f"  test {name:<22} n={m.sum():5d} correctly classified={acc_c:.4f}")

    with open(os.path.join(args.out_dir, "test_predictions.csv"), "w", encoding="utf-8") as f:
        f.write("path,category,identity,label_fake,prob_fake,pred_fake\n")
        for j, i in enumerate(test_idx):
            s = samples[i]
            f.write(f"\"{s['path']}\",{s['category']},{s['group']},{ybin[i]},"
                    f"{p_test[j]:.6f},{pred[j]}\n")
    json.dump({"threshold": thr, "threshold_selection": info, "test_ensemble": rep,
               "folds": fold_rows, "args": vars(args)},
              open(os.path.join(args.out_dir, "results.json"), "w"), indent=2)
    log(f"Saved: {args.out_dir}/results.json, test_predictions.csv, avff_v3_fold*.pth")
    log_f.close()


if __name__ == "__main__":
    main()
