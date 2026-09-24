"""Bidirectional selective state-space model on the same causal information as the trees.

Outage channels are zero after the cutoff. The backward scan sees future weather, which
the rules allow, and zeros for future outages. Final weights are fit on every training
county for the median epoch chosen by severity-stratified early stopping.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedShuffleSplit

from night_owls.config import BASE_SEED, LEAD, OBSERVED, SEQ_CONFIG
from night_owls.cv import make_folds
from night_owls.features import weather_transform
from night_owls.io import CUTOFF_TS, read_frame, validate_frame
from night_owls.metrics import horizon_scores, horizon_weights
from night_owls.prior import estimate_half_life

TARGET_SCALE = 10.0


def build_sequences(df):
    """(n, 216, 35) float32 sequences. Outage channels are zero after the cutoff.

    Reconstructed OSI is rounded to 4 decimals so it matches the official series.
    """
    seqs, y, last, fips = [], [], [], []
    for county, g0 in df.groupby("fipsCode", sort=True):
        g = g0.reset_index(drop=True)
        w = weather_transform(g).to_numpy(dtype=np.float32)
        observed = (g.timestamp_et <= CUTOFF_TS).to_numpy()
        o = g[["P_t", "N_t", "D_t", "R_t"]].to_numpy(dtype=np.float64)
        o[~observed] = 0.0
        osi = np.round(np.maximum(0.40 * o[:, 0] + 0.35 * o[:, 1] + 0.25 * o[:, 2] - 0.10 * o[:, 3], 0), 4)
        outage = np.c_[o, osi] * 10.0
        hour = g.timestamp_et.dt.hour.to_numpy()
        t = np.arange(216)
        extra = np.c_[
            observed.astype(np.float32),
            np.sin(2 * np.pi * hour / 24),
            np.cos(2 * np.pi * hour / 24),
            np.maximum(t - 71, 0) / 144.0,
            np.full(216, np.log1p(g.customersTracked.iloc[:72].median()) / 12.0),
        ]
        seqs.append(np.c_[w, outage, extra].astype(np.float32))
        if "osi" in g:
            y.append(g.osi.to_numpy(dtype=np.float32)[72:])
        else:
            y.append(np.full(144, np.nan, dtype=np.float32))
        last.append(float(osi[71]))
        fips.append(int(county))
    return np.stack(seqs), np.stack(y), np.asarray(last), np.asarray(fips)


def sequence_leakage_test(df, X, last):
    """Scrambling post-cutoff outages, whole-event columns, and future customer counts must not change inputs."""
    poisoned = df.copy()
    future = poisoned.timestamp_et > CUTOFF_TS
    forbidden = [
        c for c in poisoned
        if c in list(OBSERVED) + ["osi", "customersTracked"]
        or str(c).startswith(("osi_target_", "osi_delta_", "osi_lag", "outage_pct_lag"))
    ]
    poisoned.loc[future, forbidden] = 987654
    for c in ["peak_pct", "peak_customers", "time_to_restore_h", "event_duration_h", "severity_tier"]:
        if c in poisoned:
            poisoned[c] = 987654.321
    X2, _, last2, _ = build_sequences(poisoned)
    if not (np.array_equal(X, X2) and np.array_equal(last, last2)):
        raise AssertionError("Forbidden post-cutoff data changed the sequence inputs.")


def with_prior_channel(X, prior):
    ch = np.zeros((len(X), 216, 1), dtype=np.float32)
    ch[:, 72:, 0] = prior * 10.0
    return np.concatenate([X, ch], axis=-1)


def chunked_scan(dA, dBx, chunk=4):
    """h_t = dA_t h_{t-1} + dBx_t, exactly, with a closed form inside chunks."""
    h = torch.zeros(dA.shape[0], dA.shape[2], dA.shape[3], dtype=dA.dtype)
    out = []
    for c0 in range(0, dA.shape[1], chunk):
        a, b = dA[:, c0:c0 + chunk], dBx[:, c0:c0 + chunk]
        P = torch.cumprod(a, dim=1)
        hc = P * (torch.cumsum(b / P, dim=1) + h.unsqueeze(1))
        out.append(hc)
        h = hc[:, -1]
    return torch.cat(out, dim=1)


class S6Block(nn.Module):
    """Mamba block: gated selective SSM with input-dependent B, C, and step size."""

    def __init__(self, d_model, d_state=8, d_conv=4, expand=2):
        super().__init__()
        self.d_inner, self.d_state = expand * d_model, d_state
        self.dt_rank = math.ceil(d_model / 16)
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)
        self.conv = nn.Conv1d(self.d_inner, self.d_inner, d_conv, groups=self.d_inner, padding=d_conv - 1)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner)
        with torch.no_grad():
            dt = torch.exp(torch.rand(self.d_inner) * (math.log(0.1) - math.log(1e-3)) + math.log(1e-3))
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1, dtype=torch.float32)).repeat(self.d_inner, 1))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, u):
        length = u.shape[1]
        x, z = self.in_proj(u).chunk(2, dim=-1)
        x = F.silu(self.conv(x.transpose(1, 2))[:, :, :length].transpose(1, 2))
        dt, Bm, Cm = torch.split(self.x_proj(x), [self.dt_rank, self.d_state, self.d_state], dim=-1)
        delta = F.softplus(self.dt_proj(dt))
        dA = torch.exp(delta.unsqueeze(-1) * (-torch.exp(self.A_log)))
        dBx = delta.unsqueeze(-1) * Bm.unsqueeze(2) * x.unsqueeze(-1)
        y = (chunked_scan(dA, dBx) * Cm.unsqueeze(2)).sum(-1) + x * self.D
        return self.out_proj(y * F.silu(z))


class BiMamba(nn.Module):
    """Two bidirectional S6 layers. The backward scan reads future weather only."""

    def __init__(self, n_in, d_model=24, n_layers=2, d_state=8, d_conv=4, expand=2, dropout=0.1, **_):
        super().__init__()
        self.embed = nn.Linear(n_in, d_model)
        self.fwd = nn.ModuleList([S6Block(d_model, d_state, d_conv, expand) for _ in range(n_layers)])
        self.bwd = nn.ModuleList([S6Block(d_model, d_state, d_conv, expand) for _ in range(n_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.drop = nn.Dropout(dropout)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 1))

    def forward(self, x):
        h = self.embed(x)
        for fwd, bwd, norm in zip(self.fwd, self.bwd, self.norms):
            hn = norm(h)
            h = h + self.drop(fwd(hn) + bwd(hn.flip(1)).flip(1))
        return self.head(h)[:, 72:, 0]


def _es_split(n: int, seed: int, fraction: float, strata):
    """Return (train_rows, early_stop_rows). Without strata this matches the original permutation split."""
    if strata is not None:
        try:
            splitter = StratifiedShuffleSplit(n_splits=1, test_size=fraction, random_state=seed)
            train_rows, es_rows = next(splitter.split(np.zeros(n), strata))
            if len(train_rows) >= 1 and len(es_rows) >= 1:
                return train_rows, es_rows
        except ValueError:
            pass
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_es = max(int(fraction * n), 20)
    n_es = min(n_es, n - 1)
    return idx[n_es:], idx[:n_es]


def fit_predict(Xtr, ytr, Xq, seed, cfg=None, strata=None, fixed_epochs=None):
    """Train on Xtr/ytr and predict Xq.

    With fixed_epochs, every row of Xtr is used for that many epochs (final fit).
    Otherwise a stratified early-stopping split selects the epoch.
    """
    cfg = dict(SEQ_CONFIG if cfg is None else cfg)
    torch.manual_seed(seed)
    if fixed_epochs is None:
        train_rows, es_rows = _es_split(len(Xtr), seed, cfg["es_fraction"], strata)
    else:
        train_rows, es_rows = np.arange(len(Xtr)), None
    flat = Xtr[train_rows].reshape(-1, Xtr.shape[-1])
    mu, sd = flat.mean(0), flat.std(0) + 1e-6

    def norm(a):
        return torch.tensor((a - mu) / sd, dtype=torch.float32)

    Xt = norm(Xtr[train_rows])
    yt = torch.tensor(ytr[train_rows] * TARGET_SCALE)
    Xe = ye = None
    if es_rows is not None:
        Xe, ye = norm(Xtr[es_rows]), torch.tensor(ytr[es_rows] * TARGET_SCALE)
    Xq_ = norm(Xq)
    w = torch.tensor(horizon_weights(normalize=True), dtype=torch.float32)
    model = BiMamba(Xtr.shape[-1], **cfg)
    epochs = int(fixed_epochs if fixed_epochs is not None else cfg["max_epochs"])
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best, best_state, bad, used = np.inf, None, 0, 0
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), cfg["batch_size"]):
            b = perm[i:i + cfg["batch_size"]]
            loss = ((model(Xt[b]) - yt[b]) ** 2 * w).sum(1).mean()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        used = epoch + 1
        if Xe is None:
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            continue
        model.eval()
        with torch.no_grad():
            ve = float(((model(Xe) - ye) ** 2 * w).sum(1).mean())
        if ve < best - 1e-7:
            best, bad, best_state = ve, 0, {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg["patience"]:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        return np.maximum(model(Xq_).numpy() / TARGET_SCALE, 0), used


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--seed", type=int, default=BASE_SEED)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--no-cv", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    started = time.time()
    train = read_frame(args.data_dir / "DM_Train.csv")
    test = read_frame(args.data_dir / "DM_Test.csv")
    validate_frame(train, False)
    validate_frame(test, True)
    X, y, last, fips = build_sequences(train)
    Xt, _, last_t, fips_t = build_sequences(test)
    sequence_leakage_test(train, X, last)
    print("Sequence leakage mutation test passed.", flush=True)
    county = train.drop_duplicates("fipsCode").set_index("fipsCode").loc[fips].reset_index()
    strata = (county.stateAbbr.astype(str) + "_" + county.severity_tier.astype(str)).to_numpy()
    manifest = {
        "config": SEQ_CONFIG, "seed": args.seed, "torch": torch.__version__,
        "n_parameters": int(sum(p.numel() for p in BiMamba(X.shape[-1] + 1).parameters())),
        "osi_rounding": "4 decimals, matching the official series",
    }
    epochs = []
    if not args.no_cv:
        oof = np.full_like(y, np.nan)
        for k, (tr, va) in enumerate(make_folds(strata, args.folds, args.seed)):
            half_life = estimate_half_life(y[tr], last[tr])
            Xk = with_prior_channel(X, last[:, None] * np.exp2(-LEAD[None, :] / half_life))
            preds = []
            for s in range(SEQ_CONFIG["seeds"]):
                pred, used = fit_predict(Xk[tr], y[tr], Xk[va], args.seed + 100 * k + s, strata=strata[tr])
                preds.append(pred)
                epochs.append(used)
                print(f"fold {k + 1}/{args.folds} seed {s}: {used} epochs, elapsed {time.time() - started:.0f}s", flush=True)
            oof[va] = np.mean(preds, axis=0)
        scores = horizon_scores(y, oof)
        manifest["oof_scores"] = scores
        manifest["epochs_per_run"] = epochs
        np.savez_compressed(args.output_dir / "sequence_oof.npz", oof=oof, truth=y, fips=fips)
        print("out-of-fold:", {k: round(v, 6) for k, v in scores.items()}, flush=True)
    half_life = estimate_half_life(y, last)
    manifest["final_half_life_h"] = half_life
    manifest["final_fit"] = "early-stopping holdout, same protocol as cross-validation"
    Xall = with_prior_channel(X, last[:, None] * np.exp2(-LEAD[None, :] / half_life))
    Xtest = with_prior_channel(Xt, last_t[:, None] * np.exp2(-LEAD[None, :] / half_life))
    preds = []
    for s in range(SEQ_CONFIG["seeds"]):
        pred, used = fit_predict(Xall, y, Xtest, args.seed + 1000 + s, strata=strata)
        preds.append(pred)
        print(f"final fit seed {s}: {used} epochs, elapsed {time.time() - started:.0f}s", flush=True)
    np.savez_compressed(args.output_dir / "sequence_test.npz", fips=fips_t, pred=np.mean(preds, axis=0))
    manifest["runtime_seconds"] = round(time.time() - started, 1)
    (args.output_dir / "sequence_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"done in {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
