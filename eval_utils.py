"""
eval_utils.py  --  torch-free helpers for leakage-safe splitting, threshold
selection and metric reporting.  Used by avff_v3_full.py.

Conventions
-----------
* Binary labels: 0 = REAL (RealVideo-RealAudio), 1 = FAKE (any manipulation).
* A probability p is the model's P(fake).  Prediction = FAKE if p >= threshold.
* "Macro" metrics average the REAL and FAKE class equally.  With a 1:42
  imbalance, accuracy / weighted / fake-class metrics are trivially high, so
  macro metrics are the honest headline numbers.
"""
import re
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedGroupKFold

CATEGORY_TO_LABEL = {
    "RealVideo-RealAudio": 0,
    "RealVideo-FakeAudio": 1,
    "FakeVideo-RealAudio": 2,
    "FakeVideo-FakeAudio": 3,
}

_ID_RE = re.compile(r"^id\d+$", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Identity-grouped splits (prevents the same person appearing in train & test)
# ---------------------------------------------------------------------------

def parse_identity(path):
    """FakeAVCeleb layout: <root>/<Category>/<Ethnicity>/<gender>/<idXXXXX>/<file>.mp4
    Returns the idXXXXX folder name; falls back to the parent directory name."""
    parts = re.split(r"[\\/]+", path)
    ids = [p for p in parts[:-1] if _ID_RE.match(p)]
    if ids:
        return ids[-1].lower()
    return parts[-2].lower() if len(parts) >= 2 else path


def make_grouped_splits(labels4, groups, test_frac=0.2, n_folds=5, seed=42):
    """Returns (dev_idx, test_idx, folds) where folds = [(train_idx, val_idx), ...]
    indexing into the ORIGINAL arrays.  Splitting is by identity and stratified
    on the 4-way category so every split contains all manipulation types."""
    labels4 = np.asarray(labels4)
    groups = np.asarray(groups)
    idx = np.arange(len(labels4))

    n_outer = max(2, int(round(1.0 / test_frac)))
    outer = StratifiedGroupKFold(n_splits=n_outer, shuffle=True, random_state=seed)
    dev_idx, test_idx = next(outer.split(idx, labels4, groups))

    inner = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = []
    for tr, va in inner.split(dev_idx, labels4[dev_idx], groups[dev_idx]):
        folds.append((dev_idx[tr], dev_idx[va]))

    # Sanity: no identity overlap between test and dev, nor between train/val
    assert not (set(groups[dev_idx]) & set(groups[test_idx])), "identity leak dev/test"
    for tr, va in folds:
        assert not (set(groups[tr]) & set(groups[va])), "identity leak train/val"
    return dev_idx, test_idx, folds


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _counts(y, pred, w=None):
    y = np.asarray(y).astype(int)
    pred = np.asarray(pred).astype(int)
    w = np.ones(len(y)) if w is None else np.asarray(w, dtype=float)
    tp = w[(y == 1) & (pred == 1)].sum()   # fake caught
    tn = w[(y == 0) & (pred == 0)].sum()   # real kept
    fp = w[(y == 0) & (pred == 1)].sum()   # real called fake
    fn = w[(y == 1) & (pred == 0)].sum()   # fake called real
    return tp, tn, fp, fn


def _safe(a, b):
    return float(a / b) if b > 0 else 0.0


def metrics_from_preds(y, pred, w=None):
    tp, tn, fp, fn = _counts(y, pred, w)
    p_fake, r_fake = _safe(tp, tp + fp), _safe(tp, tp + fn)
    p_real, r_real = _safe(tn, tn + fn), _safe(tn, tn + fp)
    f1_fake = _safe(2 * p_fake * r_fake, p_fake + r_fake)
    f1_real = _safe(2 * p_real * r_real, p_real + r_real)
    return {
        "accuracy": _safe(tp + tn, tp + tn + fp + fn),
        "balanced_accuracy": (r_fake + r_real) / 2,
        "macro_precision": (p_fake + p_real) / 2,
        "macro_recall": (r_fake + r_real) / 2,
        "macro_f1": (f1_fake + f1_real) / 2,
        "precision_real": p_real, "recall_real": r_real, "f1_real": f1_real,
        "precision_fake": p_fake, "recall_fake": r_fake, "f1_fake": f1_fake,
        "tp": float(tp), "tn": float(tn), "fp": float(fp), "fn": float(fn),
    }


def full_report(y, p, threshold):
    """Metrics on the natural (imbalanced) distribution AND the class-balanced
    (prior-corrected) view, where each class carries equal total weight.  The
    balanced view equals the expected metrics on a 50/50 test set and is
    deterministic (no random subsampling)."""
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    pred = (p >= threshold).astype(int)
    n0, n1 = max((y == 0).sum(), 1), max((y == 1).sum(), 1)
    w_bal = np.where(y == 0, 0.5 / n0, 0.5 / n1)
    out = {
        "threshold": float(threshold),
        "n_real": int((y == 0).sum()), "n_fake": int((y == 1).sum()),
        "auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan"),
        "ap_fake": float(average_precision_score(y, p)) if len(np.unique(y)) == 2 else float("nan"),
        "ap_real": float(average_precision_score(1 - y, 1 - p)) if len(np.unique(y)) == 2 else float("nan"),
        "natural": metrics_from_preds(y, pred),
        "class_balanced": metrics_from_preds(y, pred, w_bal),
    }
    return out


KEY4 = ("accuracy", "macro_precision", "macro_recall", "macro_f1")


def format_report(rep, title=""):
    lines = []
    if title:
        lines.append(title)
    lines.append(f"  threshold={rep['threshold']:.4f} | n_real={rep['n_real']} n_fake={rep['n_fake']} | "
                 f"AUC={rep['auc']:.4f} AP(fake)={rep['ap_fake']:.4f} AP(real)={rep['ap_real']:.4f}")
    for view in ("natural", "class_balanced"):
        m = rep[view]
        lines.append(
            f"  [{view:>14}] Acc={m['accuracy']:.4f} BalAcc={m['balanced_accuracy']:.4f} | "
            f"Macro P={m['macro_precision']:.4f} R={m['macro_recall']:.4f} F1={m['macro_f1']:.4f} | "
            f"Real P/R={m['precision_real']:.3f}/{m['recall_real']:.3f} "
            f"Fake P/R={m['precision_fake']:.3f}/{m['recall_fake']:.3f}")
    m = rep["natural"]
    lines.append(f"  confusion (natural): TN(real ok)={int(m['tn'])} FP(real->fake)={int(m['fp'])} "
                 f"FN(fake->real)={int(m['fn'])} TP(fake ok)={int(m['tp'])}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Threshold selection  (VALIDATION DATA ONLY)
# ---------------------------------------------------------------------------

def candidate_thresholds(p, n=2000):
    p = np.asarray(p, dtype=float)
    q = np.unique(np.quantile(p, np.linspace(0, 1, n)))
    mids = (q[:-1] + q[1:]) / 2 if len(q) > 1 else q
    return np.unique(np.concatenate([q, mids, [0.5]]))


def select_threshold(y, p, target=0.80, objective="macro_f1", view="natural"):
    """Pick the decision threshold on VALIDATION predictions.

    1. Feasible set = thresholds where accuracy, macro precision, macro recall
       and macro F1 are ALL >= target.
    2. If the feasible set is non-empty, choose within it by `objective`:
         'macro_f1'   -> highest macro F1   (matches the paper sentence)
         'min_metric' -> largest worst-case metric (most safety margin)
    3. If it is empty, fall back to the threshold maximising the worst of the
       four metrics and flag feasible=False (the target is NOT met on val).
    Returns (threshold, info_dict).
    """
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    w = None
    if view == "class_balanced":
        n0, n1 = max((y == 0).sum(), 1), max((y == 1).sum(), 1)
        w = np.where(y == 0, 0.5 / n0, 0.5 / n1)

    best = None
    rows = []
    for t in candidate_thresholds(p):
        m = metrics_from_preds(y, (p >= t).astype(int), w)
        vals = np.array([m[k] for k in KEY4])
        rows.append((t, vals.min(), m["macro_f1"], bool((vals >= target).all()), m))

    feas = [r for r in rows if r[3]]
    if feas:
        key = (lambda r: (r[2], r[1])) if objective == "macro_f1" else (lambda r: (r[1], r[2]))
        best = max(feas, key=key)
        lo, hi = min(r[0] for r in feas), max(r[0] for r in feas)
    else:
        best = max(rows, key=lambda r: (r[1], r[2]))
        lo = hi = float("nan")
    t, _, _, ok, m = best
    info = {"threshold": float(t), "feasible": ok, "view": view, "objective": objective,
            "target": target, "feasible_range": [float(lo), float(hi)],
            **{k: m[k] for k in KEY4}}
    return float(t), info
