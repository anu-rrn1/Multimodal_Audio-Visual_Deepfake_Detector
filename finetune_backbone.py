#!/usr/bin/env python
# coding: utf-8
"""
Fix 2: fine-tune ResNet-18 on face frames (CPU-only)
=====================================================
The frozen ImageNet ResNet-18 has never seen deepfakes, so real videos and
face-swapped videos with REAL audio (FVRA) look alike to it.  This script
fine-tunes ResNet-18 to separate REAL-video frames (RealVideo-RealAudio and
RealVideo-FakeAudio, whose video track is genuine) from FAKE-video frames
(FakeVideo-RealAudio and FakeVideo-FakeAudio).  The result is then used by
avff_v3_full.py to re-extract the video features (--backbone_ckpt).

Leakage-safe protocol (one identity-grouped split; same split as avff_v3_full.py)
  * Same identity-grouped dev/test split and 5 folds (same seed).
  * The backbone trains ONLY on identities of fold --fold's TRAIN split.
  * Epoch selection uses that fold's VALIDATION split.  Test is never touched.
  * The checkpoint stores every identity it was trained on; avff_v3_full.py
    refuses to run any fold whose validation/test identities overlap them.
On a CPU this is affordable for one split only, so AVFF is then run with
--folds <fold> instead of 5-fold cross-validation.

Steps (all resumable):
  1. Frame bank: decodes the selected frames once into uint8 memmaps (frame_bank/).
  2. Baseline: linear probe on the frozen ImageNet features (what AVFF used so far).
  3. Fine-tune layer3 + layer4 + a 2-class head; keep the epoch with the best
     validation video-level AUC.
  4. Go/no-go: fine-tuned vs ImageNet baseline on validation, before you spend
     ~5 h re-extracting features.

Usage:
  python finetune_backbone.py                 # ~2.5-4.5 h on a 2-core i3 laptop (prints its ETA)
"""

import os
import sys
import json
import math
import time
import argparse
import warnings
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import avff_v3_full as A
import eval_utils as E

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
REAL_VIDEO = (0, 1)                 # RVRA, RVFA: genuine video track


# =============================================================================
# SAMPLES + SPLIT  (identical to avff_v3_full.py)
# =============================================================================

def usable_samples(args):
    """Same filtering order as avff_v3_full.py: discover -> (debug subset) -> keep
    videos the reference cache could decode.  Guarantees the same split."""
    samples = A.discover_dataset(args.dataset_path)
    if args.max_samples and args.max_samples < len(samples):
        rng = np.random.default_rng(args.seed)
        samples = [samples[i] for i in sorted(rng.choice(len(samples), args.max_samples, False))]
        print(f"[Debug] random subset of {len(samples)} videos (NOT for reported results)")
    ok = {}
    for sf in A.shard_files(args.ref_cache):
        d = torch.load(sf, map_location="cpu")
        for i, p in enumerate(d["paths"]):
            ok[p] = bool(d["v_ok"][i])
        del d
    if not ok:
        sys.exit(f"[Error] reference cache {args.ref_cache} is empty; it is needed to reproduce "
                 f"the exact split of avff_v3_full.py")
    keep = [s for s in samples if ok.get(s["path"], False)]
    print(f"Usable videos (decodable in {os.path.basename(args.ref_cache)}): {len(keep)}")
    return keep


def spread_over_identities(idx, groups, n, rng):
    """Pick n indices round-robin over identities (diverse people, not 1 person x many)."""
    by_g = {}
    for i in rng.permutation(idx):
        by_g.setdefault(groups[i], []).append(i)
    order = list(by_g)
    rng.shuffle(order)
    out = []
    while len(out) < n and by_g:
        for g in list(order):
            if by_g.get(g):
                out.append(by_g[g].pop())
                if len(out) == n:
                    break
            else:
                by_g.pop(g, None)
                order.remove(g)
    return np.array(sorted(out), dtype=int)


# =============================================================================
# FRAME BANK  (uint8 memmap, resumable)
# =============================================================================

def pick_frames(n_have, n_want):
    return np.linspace(0, n_have - 1, n_want).round().astype(int)


class FrameBank:
    def __init__(self, folder, name, plan, size):
        """plan: list of (sample, n_frames_to_keep)."""
        self.dir = folder
        self.size = size
        self.plan = plan
        self.n = sum(k for _, k in plan)
        self.offsets = np.concatenate([[0], np.cumsum([k for _, k in plan])]).astype(int)
        self.img_path = os.path.join(folder, f"{name}_frames.u8")
        self.done_path = os.path.join(folder, f"{name}_done.npy")
        self.ok_path = os.path.join(folder, f"{name}_ok.npy")
        mode = "r+" if os.path.exists(self.img_path) else "w+"
        self.img = np.memmap(self.img_path, dtype=np.uint8, mode=mode,
                             shape=(self.n, size, size, 3))
        self.done = (np.load(self.done_path) if os.path.exists(self.done_path)
                     else np.zeros(len(plan), bool))
        self.ok = (np.load(self.ok_path) if os.path.exists(self.ok_path)
                   else np.zeros(len(plan), bool))
        # per-frame metadata
        self.vid = np.repeat(np.arange(len(plan)), [k for _, k in plan])
        self.cat = np.array([s["label"] for s, _ in plan])[self.vid]
        self.y = (~np.isin(self.cat, REAL_VIDEO)).astype(int)     # 1 = fake video

    def fill(self, args, title):
        todo = [i for i in range(len(self.plan)) if not self.done[i]]
        if not todo:
            print(f"  [Bank] {title}: {self.n} frames ready")
            return
        print(f"  [Bank] {title}: decoding {len(todo)} videos -> {self.img_path}")
        t0 = time.time()

        def job(i):
            s, k = self.plan[i]
            fr = A.read_frames(s["path"], args.n_frames, args.img_size, args.face_crop)
            return i, (None if fr is None else fr[pick_frames(len(fr), k)])

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            for j, (i, fr) in enumerate(ex.map(job, todo)):
                if fr is not None:
                    self.img[self.offsets[i]:self.offsets[i + 1]] = fr
                    self.ok[i] = True
                self.done[i] = True
                if (j + 1) % 64 == 0 or j + 1 == len(todo):
                    self.img.flush()
                    np.save(self.done_path, self.done)
                    np.save(self.ok_path, self.ok)
                    el = time.time() - t0
                    print(f"\r  [Bank] {j+1}/{len(todo)} videos | {(j+1)/el:.2f} vid/s | "
                          f"ETA {(len(todo)-j-1)/max((j+1)/el,1e-6)/60:.1f} min   ",
                          end="", flush=True)
        print()

    def valid_frames(self):
        return np.where(self.ok[self.vid])[0]


# =============================================================================
# MODEL
# =============================================================================

MEAN = torch.tensor(A.IMAGENET_MEAN).view(1, 3, 1, 1)
STD = torch.tensor(A.IMAGENET_STD).view(1, 3, 1, 1)


def to_tensor(u8):
    x = torch.from_numpy(np.ascontiguousarray(u8)).permute(0, 3, 1, 2).float().div_(255)
    return x


def augment(x, rng):
    """x: (B,3,H,W) in [0,1].  Flip, random-resized-crop (scale 0.8-1), light
    brightness/contrast jitter.  No blur / JPEG: they would erase artefacts."""
    B, _, H, W = x.shape
    out = torch.empty_like(x)
    for i in range(B):
        xi = x[i]
        if rng.random() < 0.5:
            xi = xi.flip(-1)
        s = math.sqrt(rng.uniform(0.8, 1.0))
        r = math.sqrt(rng.uniform(0.9, 1.1))
        h, w = min(H, int(H * s / r)), min(W, int(W * s * r))
        y0, x0 = rng.integers(0, H - h + 1), rng.integers(0, W - w + 1)
        xi = xi[:, y0:y0 + h, x0:x0 + w]
        if (h, w) != (H, W):
            xi = F.interpolate(xi.unsqueeze(0), size=(H, W), mode="bilinear",
                               align_corners=False).squeeze(0)
        b, c = rng.uniform(-0.08, 0.08), rng.uniform(0.85, 1.15)
        m = xi.mean()
        out[i] = ((xi - m) * c + m + b).clamp_(0, 1)
    return out


class FTNet(nn.Module):
    """ResNet-18 with conv1..layer2 frozen, layer3/layer4 trainable, 2-class head
    on the layer4 average pool (the same head avff_v3_full.py uses at extraction)."""

    def __init__(self, weights_path):
        super().__init__()
        self.net = A.ResNet18Features(weights_path).net     # strict ImageNet loading
        for name, p in self.net.named_parameters():
            p.requires_grad_(name.startswith(("layer3.", "layer4.")))
        self.head = nn.Linear(512, 2)
        self.drop = nn.Dropout(0.2)

    def train(self, mode=True):
        super().train(mode)
        self.net.eval()                  # BatchNorm keeps ImageNet statistics (small batches)
        return self

    def forward(self, x):
        n = self.net
        x = ((x - MEAN) / STD).contiguous(memory_format=torch.channels_last)
        with torch.no_grad():
            x = n.layer2(n.layer1(n.maxpool(n.relu(n.bn1(n.conv1(x))))))
        l3 = n.layer3(x)
        l4 = n.layer4(l3)
        return self.head(self.drop(l4.mean((2, 3)))), (l4, l3)


@torch.no_grad()
def frame_margins(model, bank, idx, bs=64):
    """Fake-vs-real logit margin per frame."""
    model.eval()
    out = np.zeros(len(idx), np.float32)
    for i in range(0, len(idx), bs):
        lg, _ = model(to_tensor(bank.img[idx[i:i + bs]]))
        out[i:i + bs] = (lg[:, 1] - lg[:, 0]).numpy()
    return out


def video_report(bank, idx, score):
    """Video-level AUCs (mean frame score per video)."""
    from sklearn.metrics import roc_auc_score
    vids = bank.vid[idx]
    uv = np.unique(vids)
    vs = np.array([score[vids == v].mean() for v in uv])
    cat = np.array([bank.plan[v][0]["label"] for v in uv])
    yv = (~np.isin(cat, REAL_VIDEO)).astype(int)

    def auc(mask):
        m = np.asarray(mask)
        return float(roc_auc_score(yv[m], vs[m])) if len(np.unique(yv[m])) == 2 else float("nan")

    return {"video_auc": auc(np.ones(len(uv), bool)),
            "RVRA_vs_FVRA_auc": auc(np.isin(cat, (0, 2))),
            "frame_auc": float(roc_auc_score(bank.y[idx], score)),
            "n_videos": int(len(uv)), "n_real_video": int((yv == 0).sum())}


def fmt(r):
    return (f"video AUC={r['video_auc']:.4f} | RVRA vs FVRA AUC={r['RVRA_vs_FVRA_auc']:.4f} | "
            f"frame AUC={r['frame_auc']:.4f} ({r['n_videos']} videos, "
            f"{r['n_real_video']} real-video)")


# =============================================================================
# BASELINE: linear probe on the frozen ImageNet features AVFF used so far
# =============================================================================

def imagenet_probe(args, tr_bank, va_bank, rng, log):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    ext = A.ResNet18Features(args.weights_path)
    tri = tr_bank.valid_frames()
    tri_real = rng.permutation(tri[tr_bank.y[tri] == 0])[:args.probe_frames // 2]
    tri_fake = rng.permutation(tri[tr_bank.y[tri] == 1])[:args.probe_frames // 2]
    tri = np.sort(np.concatenate([tri_real, tri_fake]))
    vai = va_bank.valid_frames()

    def feats(bank, idx):
        out = []
        for i in range(0, len(idx), 128):
            x = to_tensor(bank.img[idx[i:i + 128]])
            out.append(ext(((x - MEAN) / STD).contiguous(memory_format=torch.channels_last)))
        return torch.cat(out).numpy()

    t0 = time.time()
    Xtr, Xva = feats(tr_bank, tri), feats(va_bank, vai)
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=0.05, max_iter=3000, class_weight="balanced")
    clf.fit(sc.transform(Xtr), tr_bank.y[tri])
    r = video_report(va_bank, vai, clf.decision_function(sc.transform(Xva)))
    log(f"  [Baseline] frozen ImageNet ResNet-18 (1280-d, linear probe on {len(tri)} train "
        f"frames, {time.time()-t0:.0f}s): {fmt(r)}")
    return r


# =============================================================================
# MAIN
# =============================================================================

def main():
    ap = argparse.ArgumentParser(description="Fix 2: fine-tune ResNet-18 on face frames (CPU)")
    ap.add_argument("--dataset_path", default=A.DEFAULT_DATASET_PATH)
    ap.add_argument("--weights_path", default=A.DEFAULT_WEIGHTS_PATH)
    ap.add_argument("--ref_cache", default=os.path.join(HERE, "features_v3"),
                    help="existing feature cache; only used to reproduce the exact split")
    ap.add_argument("--bank_dir", default=os.path.join(HERE, "frame_bank"))
    ap.add_argument("--out_dir", default=os.path.join(HERE, "backbone_ft"))
    ap.add_argument("--max_samples", type=int, default=0, help="debug subset (as in avff_v3_full)")
    ap.add_argument("--fold", type=int, default=1, help="fold whose TRAIN split is used (1-based)")
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--test_frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    # frames (must match extraction)
    ap.add_argument("--n_frames", type=int, default=16)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--face_crop", default="auto", choices=["auto", "on", "off"])
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 4)))
    # bank composition
    ap.add_argument("--fake_frames", type=int, default=8, help="frames per fake-video train video")
    ap.add_argument("--val_real_frames", type=int, default=4)
    ap.add_argument("--val_fake_videos", type=int, default=640)
    ap.add_argument("--val_fake_frames", type=int, default=2)
    # training
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--steps", type=int, default=250, help="steps per epoch")
    ap.add_argument("--batch_size", type=int, default=32, help="half real-video, half fake-video")
    ap.add_argument("--lr", type=float, default=1e-4, help="layer3/layer4")
    ap.add_argument("--lr_head", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--label_smoothing", type=float, default=0.1)
    ap.add_argument("--probe_frames", type=int, default=4000)
    ap.add_argument("--no_probe", action="store_true", help="skip the ImageNet baseline")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    torch.set_num_threads(args.threads)
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.bank_dir, exist_ok=True)
    log_f = open(os.path.join(args.out_dir, f"finetune_log_fold{args.fold}.txt"), "a",
                 encoding="utf-8")

    def log(msg=""):
        print(msg)
        log_f.write(msg + "\n")
        log_f.flush()

    log("=" * 78)
    log(f"Fix 2 backbone fine-tuning | {time.strftime('%Y-%m-%d %H:%M:%S')} | "
        f"CPU threads={args.threads}")
    log("Args: " + json.dumps(vars(args)))

    # ---------------- split (identical to avff_v3_full.py) --------------------
    samples = usable_samples(args)
    y4 = np.array([s["label"] for s in samples])
    groups = np.array([s["group"] for s in samples])
    dev_idx, test_idx, folds = E.make_grouped_splits(y4, groups, args.test_frac, args.n_folds,
                                                     args.seed)
    if not 1 <= args.fold <= len(folds):
        sys.exit(f"[Error] --fold must be 1..{len(folds)}")
    tr, va = folds[args.fold - 1]
    assert not (set(groups[tr]) & (set(groups[va]) | set(groups[test_idx])))
    log(f"Fold {args.fold}: train={len(tr)} videos ({len(set(groups[tr]))} identities) | "
        f"val={len(va)} | test={len(test_idx)} (never used here)")

    # ---------------- frame bank ----------------------------------------------
    srng = np.random.default_rng(args.seed + 1000 * args.fold)
    tr_real = tr[np.isin(y4[tr], REAL_VIDEO)]
    tr_fake_all = tr[~np.isin(y4[tr], REAL_VIDEO)]
    n_fake_vid = min(len(tr_fake_all),
                     math.ceil(len(tr_real) * args.n_frames / args.fake_frames))
    tr_fake = spread_over_identities(tr_fake_all, groups, n_fake_vid, srng)
    va_real = va[np.isin(y4[va], REAL_VIDEO)]
    va_fake = spread_over_identities(va[~np.isin(y4[va], REAL_VIDEO)], groups,
                                     args.val_fake_videos, srng)
    tr_plan = ([(samples[i], args.n_frames) for i in tr_real] +
               [(samples[i], args.fake_frames) for i in tr_fake])
    va_plan = ([(samples[i], args.val_real_frames) for i in va_real] +
               [(samples[i], args.val_fake_frames) for i in va_fake])

    bank_cfg = {"fold": args.fold, "seed": args.seed, "n_folds": args.n_folds,
                "test_frac": args.test_frac, "max_samples": args.max_samples,
                "n_frames": args.n_frames, "img_size": args.img_size, "face_crop": args.face_crop,
                "train": [s["path"] for s, _ in tr_plan], "val": [s["path"] for s, _ in va_plan],
                "k_train": [k for _, k in tr_plan], "k_val": [k for _, k in va_plan]}
    cfg_path = os.path.join(args.bank_dir, "bank_config.json")
    if os.path.exists(cfg_path) and json.load(open(cfg_path)) != bank_cfg:
        sys.exit(f"[Error] {args.bank_dir} was built with different settings; delete it or use "
                 f"another --bank_dir.")
    json.dump(bank_cfg, open(cfg_path, "w"))
    gb = (sum(k for _, k in tr_plan) + sum(k for _, k in va_plan)) * args.img_size ** 2 * 3 / 1e9
    log(f"Frame bank: train {len(tr_real)} real-video videos x {args.n_frames} + {len(tr_fake)} "
        f"fake-video videos x {args.fake_frames} | val {len(va_real)} x {args.val_real_frames} + "
        f"{len(va_fake)} x {args.val_fake_frames} | {gb:.1f} GB on disk")
    tr_bank = FrameBank(args.bank_dir, f"fold{args.fold}_train", tr_plan, args.img_size)
    va_bank = FrameBank(args.bank_dir, f"fold{args.fold}_val", va_plan, args.img_size)
    tr_bank.fill(args, "train")
    va_bank.fill(args, "val")
    tri, vai = tr_bank.valid_frames(), va_bank.valid_frames()
    tri_real, tri_fake = tri[tr_bank.y[tri] == 0], tri[tr_bank.y[tri] == 1]
    log(f"  train frames: {len(tri_real)} real-video / {len(tri_fake)} fake-video | "
        f"val frames: {len(vai)}")

    # ---------------- baseline ------------------------------------------------
    state_path = os.path.join(args.out_dir, f"state_fold{args.fold}.json")
    state = json.load(open(state_path)) if os.path.exists(state_path) else {}
    if not args.no_probe and "baseline" not in state:
        state["baseline"] = imagenet_probe(args, tr_bank, va_bank, rng, log)
        json.dump(state, open(state_path, "w"), indent=2)
    elif "baseline" in state:
        log(f"  [Baseline] frozen ImageNet ResNet-18 (cached): {fmt(state['baseline'])}")

    # ---------------- fine-tuning ---------------------------------------------
    model = FTNet(args.weights_path)
    bb_params = [p for p in model.net.parameters() if p.requires_grad]
    opt = torch.optim.AdamW([{"params": bb_params, "lr": args.lr},
                             {"params": model.head.parameters(), "lr": args.lr_head}],
                            weight_decay=args.weight_decay)
    total = args.epochs * args.steps
    sch = torch.optim.lr_scheduler.LambdaLR(opt, A.lr_lambda(total, max(1, total // 20)))
    ce = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    last_path = os.path.join(args.out_dir, f"last_fold{args.fold}.pth")
    best_path = os.path.join(args.out_dir, f"resnet18_ft_fold{args.fold}.pth")
    start_ep, best = 0, -1.0
    if os.path.exists(last_path):
        ck = torch.load(last_path, map_location="cpu")
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sch.load_state_dict(ck["sch"])
        start_ep, best = ck["epoch"], ck["best"]
        log(f"  [Resume] continuing after epoch {start_ep} (best val video AUC={best:.4f})")

    half = args.batch_size // 2
    t_start, n_img = time.time(), 0
    for ep in range(start_ep, args.epochs):
        model.train()
        brng = np.random.default_rng(args.seed + 7919 * (ep + 1))
        tl, t_ep = 0.0, time.time()
        for st in range(args.steps):
            idx = np.sort(np.concatenate([brng.choice(tri_real, half),
                                          brng.choice(tri_fake, args.batch_size - half)]))
            x = augment(to_tensor(tr_bank.img[idx]), brng)
            y = torch.from_numpy(tr_bank.y[idx]).long()
            lg, _ = model(x)
            loss = ce(lg, y)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(bb_params + list(model.head.parameters()), 1.0)
            opt.step()
            sch.step()
            tl += loss.item()
            n_img += len(idx)
            if st % 10 == 9 or st == args.steps - 1:
                rate = n_img / (time.time() - t_start)
                left = ((args.epochs - ep) * args.steps - st - 1) * args.batch_size / rate
                print(f"\r    ep {ep+1}/{args.epochs} step {st+1}/{args.steps} | "
                      f"loss={tl/(st+1):.4f} | {rate:.1f} img/s | "
                      f"training ETA {left/3600:.2f} h (+ ~1-2 min val/epoch)   ",
                      end="", flush=True)
        print()
        r = video_report(va_bank, vai, frame_margins(model, va_bank, vai))
        improved = r["video_auc"] > best
        log(f"  Epoch {ep+1:02d}/{args.epochs} | loss={tl/args.steps:.4f} | "
            f"{(time.time()-t_ep)/60:.1f} min | val {fmt(r)}{'  <- best' if improved else ''}")
        if improved:
            best = r["video_auc"]
            meta = {"fold": args.fold, "seed": args.seed, "n_folds": args.n_folds,
                    "test_frac": args.test_frac, "max_samples": args.max_samples,
                    "epoch": ep + 1, "val": r,
                    "train_groups": sorted(set(groups[tr].tolist()))}
            torch.save({"backbone": model.net.state_dict(), "head": model.head.state_dict(),
                        "meta": meta}, best_path + ".tmp")
            os.replace(best_path + ".tmp", best_path)
            state["best"] = r
            state["best_epoch"] = ep + 1
            json.dump(state, open(state_path, "w"), indent=2)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sch": sch.state_dict(),
                    "epoch": ep + 1, "best": best}, last_path + ".tmp")
        os.replace(last_path + ".tmp", last_path)

    # ---------------- go / no-go ------------------------------------------------
    log("=" * 78)
    if "best" not in state:
        sys.exit("[Error] no checkpoint was saved")
    log(f"Best epoch {state['best_epoch']}: {fmt(state['best'])}")
    log(f"Saved: {best_path}")
    if "baseline" in state:
        b, f = state["baseline"], state["best"]
        log(f"Validation, fine-tuned vs frozen ImageNet: video AUC {f['video_auc']:.4f} vs "
            f"{b['video_auc']:.4f} | RVRA vs FVRA AUC {f['RVRA_vs_FVRA_auc']:.4f} vs "
            f"{b['RVRA_vs_FVRA_auc']:.4f}")
        if f["video_auc"] >= b["video_auc"] + 0.02:
            log("GO: fine-tuning clearly helps. Next: re-extract features with --backbone_ckpt.")
        else:
            log("NO-GO: fine-tuning gave < +0.02 video AUC on validation. Re-extracting features "
                "(~5 h) is unlikely to pay off; send this log before continuing.")
    log_f.close()


if __name__ == "__main__":
    main()
