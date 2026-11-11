#!/usr/bin/env python3
"""
Cluster-weight calibration and all related analyses of the manuscript.

Input
  data/image_level_QC.xlsx   one row per view (output of 05_build_features_QC.py)
  data/cluster_split.csv     YOLO split of each cluster (output of find_cluster_split.py; optional)

What it computes (tables and figures of the manuscript)
  Quality control summary ........................... Table S7
  In-sample fits on the dataset without QC .......... Figure 5
  Influence diagnostics (dataset without QC) ........ Table S2
  LOOCV calibration (QC and no QC) .................. Table 5
  Bootstrap 95% confidence intervals (QC) ........... Table 6
  Paired bootstrap comparisons + Holm correction .... Table S3
  Fits on the QC dataset ............................ Figure 7
  Aggregation analysis with null baseline ........... Figure 8
  Ellipsoid sensitivity analysis .................... Table S6
  Number of views per cluster ....................... Figure S1
  Seen vs. unseen clusters (YOLO split) ............. Table S8
  Model form: screening, power law, lasso ........... Table S9
  Leave-one-orchard-out validation .................. Table S10

Outputs
  results/calibration_results.xlsx   one sheet per table
  results/figures/                   Figures 5, 7, 8 and S1 (PNG 300 dpi and PDF)
"""

# =============================== CONFIGURATION ===============================
IMAGE_LEVEL = "data/image_level_QC.xlsx"
CLUSTER_SPLIT = "data/cluster_split.csv"       # set to None if not available
OUT_DIR = "results"

MAX_DT_MS = 100            # QC rule 1: max RGB-depth time offset
MIN_VALID_COVERAGE = 0.50  # QC rule 2: min valid-depth coverage of the mask
FX, FY = 608.0819702148438, 608.1353759765625   # focal lengths (px)

B_BOOT = 10_000            # bootstrap resamples
AGG_DRAWS = 20_000         # Monte Carlo groups per group size
VIEW_REPS = 200            # repetitions per number of views
VIEW_K = [1, 2, 3, 5, 10, 15, 20, 30]
SEED = 42
TIMEZONE = "Asia/Bangkok"  # used to derive acquisition dates (one orchard per date)
# ==============================================================================

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import OLSInfluence
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

warnings.filterwarnings("ignore")
os.makedirs(os.path.join(OUT_DIR, "figures"), exist_ok=True)

DESC = {"area": "Projected area", "voxel": "Voxel volume",
        "hull": "Convex hull volume", "dsa": "Depth-scaled projected area"}
DEPTH_BASED = ["voxel", "hull", "dsa"]
PAIRS = [("dsa", "voxel"), ("dsa", "hull"), ("dsa", "area"),
         ("voxel", "hull"), ("voxel", "area"), ("hull", "area")]
FEATURES = {"area": "Area", "width": "Width", "height": "Height", "aspect": "Aspect_Ratio",
            "density": "Density", "dmean": "Depth_Mean", "dstd": "Depth_Std",
            "voxel": "Voxel_Volume", "hull": "Convex_Hull_Volume", "dsa": "DSA"}
tables = {}


# ------------------------------------------------------------------ data
def timestamp_ms(name):
    """'_Color_1770708197709.677....png' -> 1770708197709.677 (epoch ms)."""
    return float(os.path.basename(str(name)).split("_")[-1].rsplit(".", 1)[0])


views = pd.read_excel(IMAGE_LEVEL)
views["dt_ms"] = (views["RGB"].map(timestamp_ms) - views["Depth"].map(timestamp_ms)).abs()
views["pass_dt"] = views["dt_ms"] <= MAX_DT_MS
views["pass_cov"] = views["valid_cov"] >= MIN_VALID_COVERAGE
views["geom_ok"] = views["status"] == "ok"
views["pass_qc"] = views["geom_ok"] & views["pass_dt"] & views["pass_cov"]
views["DSA"] = views["Area"] * views["Depth_Mean"] ** 2 / (FX * FY)      # m²
views["ELL"] = np.pi / 6 * (views["Width"] * views["Depth_Mean"] / FX) ** 2 \
    * (views["Height"] * views["Depth_Mean"] / FY)                        # m³
views["time"] = pd.to_datetime(views["RGB"].map(timestamp_ms), unit="ms", utc=True).dt.tz_convert(TIMEZONE)


def cluster_table(v):
    agg = {k: (c, "mean") for k, c in FEATURES.items()}
    agg.update(Weight=("Weight", "first"), n_views=("RGB", "size"), ell=("ELL", "mean"),
               date=("time", lambda t: str(t.min().date())))
    return v.groupby("Cluster_ID").agg(**agg)


noqc = cluster_table(views[views["geom_ok"]])      # complete dataset, no QC (n = 51)
qc = cluster_table(views[views["pass_qc"]])        # quality-controlled dataset (primary)
rng = np.random.default_rng(SEED)

# QC summary (Table S7)
geo = views[views["geom_ok"]]
ids = set(views["Cluster_ID"])
lost = lambda m: len(ids - set(geo.loc[m, "Cluster_ID"]))
tables["S7_QC_summary"] = pd.DataFrame([
    ["Views extracted", len(views), "", ""],
    ["Valid after geometric filters", len(geo), len(views) - len(geo), len(ids) - geo["Cluster_ID"].nunique()],
    [f"Rule 1: RGB-depth offset <= {MAX_DT_MS} ms", int(geo["pass_dt"].sum()), int((~geo["pass_dt"]).sum()), lost(geo["pass_dt"])],
    [f"Rule 2: valid-depth coverage >= {MIN_VALID_COVERAGE:.0%}", int(geo["pass_cov"].sum()), int((~geo["pass_cov"]).sum()), lost(geo["pass_cov"])],
    ["Both rules (QC dataset)", int(views["pass_qc"].sum()), int(len(geo) - views["pass_qc"].sum()), len(ids) - len(qc)],
], columns=["Step", "Views kept", "Views removed", "Clusters excluded"])


# ------------------------------------------------------------------ helpers
def loo_predict(x, y, log=False):
    n = len(y); p = np.empty(n); k = np.empty(n)
    for i in range(n):
        m = np.arange(n) != i
        if log:
            b1, b0 = np.polyfit(np.log(x[m]), np.log(y[m]), 1)
            smear = np.mean(np.exp(np.log(y[m]) - (b1 * np.log(x[m]) + b0)))   # Duan (1983)
            p[i] = np.exp(b1 * np.log(x[i]) + b0) * smear
        else:
            b1, b0 = np.polyfit(x[m], y[m], 1); p[i] = b1 * x[i] + b0
        k[i] = b1
    return p, k


def metrics(y, p):
    e = p - y
    s, b = np.polyfit(y, p, 1)
    return dict(MAE=np.abs(e).mean(), RMSE=np.sqrt((e ** 2).mean()), MAPE=100 * np.abs(e / y).mean(),
                R2=1 - (e ** 2).sum() / ((y - y.mean()) ** 2).sum(),
                relMAE=100 * np.abs(e).mean() / y.mean(), obs_pred_slope=s, obs_pred_intercept=b)


def boot_stats(y, p, idx):
    yb, pb = y[idx], p[idx]; e = pb - yb
    return (np.abs(e).mean(1), np.sqrt((e ** 2).mean(1)),
            1 - (e ** 2).sum(1) / ((yb - yb.mean(1, keepdims=True)) ** 2).sum(1))


def p_two_sided(d):
    return min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); adj = np.empty_like(p); run = 0
    for r, i in enumerate(o):
        run = max(run, (len(p) - r) * p[i]); adj[i] = min(1, run)
    return adj


# ------------------------------------------------------------------ calibration
pred, rows5 = {}, []
for name, data in [("QC", qc), ("no QC", noqc)]:
    y = data["Weight"].values
    for d in DESC:
        p, k = loo_predict(data[d].values, y)
        pred[(name, d)] = p
        rows5.append(dict(Dataset=name, Descriptor=DESC[d], n=len(y), **metrics(y, p),
                          k_mean=k.mean(), k_sd=k.std()))
tables["T5_LOOCV"] = pd.DataFrame(rows5)

y_qc = qc["Weight"].values
idx_qc = rng.integers(0, len(y_qc), (B_BOOT, len(y_qc)))
rows6 = []
for d in DESC:
    m, r, r2 = boot_stats(y_qc, pred[("QC", d)], idx_qc)
    rows6.append(dict(Descriptor=DESC[d], n=len(y_qc),
                      MAE_CI=np.percentile(m, [2.5, 97.5]).round(3).tolist(),
                      RMSE_CI=np.percentile(r, [2.5, 97.5]).round(3).tolist(),
                      R2_CI=np.percentile(r2, [2.5, 97.5]).round(3).tolist()))
tables["T6_bootstrap_CI"] = pd.DataFrame(rows6)

rowsS3 = []
for name, data in [("full (no QC)", noqc), ("quality-controlled", qc)]:
    y = data["Weight"].values; key = "QC" if name.startswith("quality") else "no QC"
    idx = rng.integers(0, len(y), (B_BOOT, len(y)))
    B = {d: boot_stats(y, pred[(key, d)], idx) for d in DESC}
    block = []
    for a, b in PAIRS:
        dm = B[a][0] - B[b][0]; dr = B[a][2] - B[b][2]
        ma, mb = metrics(y, pred[(key, a)]), metrics(y, pred[(key, b)])
        block.append(dict(Comparison=f"{DESC[a]} vs {DESC[b]}", Base=name, n=len(y),
                          dMAE=ma["MAE"] - mb["MAE"], dMAE_CI=np.percentile(dm, [2.5, 97.5]).round(3).tolist(),
                          p_MAE=p_two_sided(dm), dR2=ma["R2"] - mb["R2"],
                          dR2_CI=np.percentile(dr, [2.5, 97.5]).round(3).tolist(), p_R2=p_two_sided(dr)))
    adj = holm([r["p_MAE"] for r in block] + [r["p_R2"] for r in block])
    for i, r in enumerate(block):
        r["p_MAE_Holm"], r["p_R2_Holm"] = adj[i], adj[i + len(block)]
    rowsS3 += block
tables["S3_paired_bootstrap"] = pd.DataFrame(rowsS3)

# influence diagnostics on the dataset without QC (Table S2)
rowsS2 = []
for d in DESC:
    X = sm.add_constant(noqc[d].values); fit = sm.OLS(noqc["Weight"].values, X).fit()
    inf = OLSInfluence(fit); n, p_ = len(noqc), 2
    cook = noqc.index[inf.cooks_distance[0] > 4 / n].tolist()
    stud = noqc.index[np.abs(inf.resid_studentized_external) > 3].tolist()
    lev = noqc.index[inf.hat_matrix_diag > 3 * p_ / n].tolist()
    rowsS2.append(dict(Descriptor=DESC[d], Cook=cook, Studentized=stud, Leverage=lev,
                       Union=sorted(set(cook) | set(stud) | set(lev)),
                       in_sample_R2=fit.rsquared))
tables["S2_influence"] = pd.DataFrame(rowsS2)

# ------------------------------------------------------------------ aggregation (Figure 8)
def agg_curve(y, p):
    n = len(y); out = []
    for g in range(1, n):
        idx = np.argsort(rng.random((AGG_DRAWS, n)), axis=1)[:, :g]
        err = 100 * np.abs(p[idx].sum(1) - y[idx].sum(1)) / y[idx].sum(1)
        out.append((g, err.mean(), np.percentile(err, 5), np.percentile(err, 95)))
    return pd.DataFrame(out, columns=["clusters", "mean", "p5", "p95"])


def threshold(c, t):
    ok = (c["p95"] < t).values
    return next(int(c["clusters"].iloc[i]) for i in range(len(ok)) if ok[i:].all())


null = np.array([np.delete(y_qc, i).mean() for i in range(len(y_qc))])
curves = {"voxel": agg_curve(y_qc, pred[("QC", "voxel")]), "dsa": agg_curve(y_qc, pred[("QC", "dsa")]),
          "null": agg_curve(y_qc, null)}
tables["F8_aggregation_thresholds"] = pd.DataFrame(
    [dict(Predictor=k, clusters_20pct=threshold(c, 20), clusters_10pct=threshold(c, 10),
          single_cluster_mean=c["mean"].iloc[0]) for k, c in curves.items()])
tables["F8_aggregation_curves"] = pd.concat(curves, axis=1)

# ------------------------------------------------------------------ ellipsoid (Table S6)
rowsS6 = []
for name, data in [("full (no QC)", noqc), ("QC", qc)]:
    y = data["Weight"].values; p, _ = loo_predict(data["ell"].values, y)
    rowsS6.append(dict(Dataset=name, n=len(y), **metrics(y, p),
                       r_with_dsa=np.corrcoef(data["ell"], data["dsa"])[0, 1]))
    idx = rng.integers(0, len(y), (B_BOOT, len(y))); Be = boot_stats(y, p, idx)
    for d in DESC:
        key = "QC" if name == "QC" else "no QC"; Bd = boot_stats(y, pred[(key, d)], idx)
        rowsS6.append(dict(Dataset=name, Comparison=f"Ellipsoid vs {DESC[d]}",
                           dMAE=np.abs(p - y).mean() - np.abs(pred[(key, d)] - y).mean(),
                           p_MAE=p_two_sided(Be[0] - Bd[0]), p_R2=p_two_sided(Be[2] - Bd[2])))
tables["S6_ellipsoid"] = pd.DataFrame(rowsS6)

# ------------------------------------------------------------------ number of views (Figure S1)
qc_views = views[views["pass_qc"]]
groups = {c: g for c, g in qc_views.groupby("Cluster_ID")}
cols = {"area": "Area", "voxel": "Voxel_Volume", "hull": "Convex_Hull_Volume", "dsa": "DSA"}
rowsS1 = []
for k in VIEW_K:
    res = {d: [] for d in cols}
    for _ in range(VIEW_REPS):
        X = {d: [] for d in cols}; Y = []
        for c, g in groups.items():
            s = g.iloc[rng.choice(len(g), min(k, len(g)), replace=False)]
            for d, col in cols.items():
                X[d].append(s[col].mean())
            Y.append(g["Weight"].iloc[0])
        Y = np.array(Y)
        for d in cols:
            res[d].append(metrics(Y, loo_predict(np.array(X[d]), Y)[0])["R2"])
    rowsS1.append(dict(views=k, **{f"{d}_mean": np.mean(v) for d, v in res.items()},
                       **{f"{d}_sd": np.std(v) for d, v in res.items()}))
tables["S1_views"] = pd.DataFrame(rowsS1)

# ------------------------------------------------------------------ seen vs unseen (Table S8)
if CLUSTER_SPLIT and os.path.isfile(CLUSTER_SPLIT):
    split = pd.read_csv(CLUSTER_SPLIT).set_index("Cluster_ID")["Split"]
    seen = qc.index.map(lambda c: split.get(c) == "train").values.astype(bool)
    rowsS8 = []
    for d in DESC:
        p = pred[("QC", d)]
        u = qc[~seen]; pu, _ = loo_predict(u[d].values, u["Weight"].values)
        rowsS8.append(dict(Descriptor=DESC[d],
                           **{f"train_{k}": v for k, v in metrics(y_qc[seen], p[seen]).items() if k in ("R2", "MAE", "MAPE")},
                           **{f"unseen_{k}": v for k, v in metrics(y_qc[~seen], p[~seen]).items() if k in ("R2", "MAE", "MAPE")},
                           **{f"unseen_recal_{k}": v for k, v in metrics(u["Weight"].values, pu).items() if k in ("R2", "MAE", "MAPE")},
                           n_train=int(seen.sum()), n_unseen=int((~seen).sum())))
    tables["S8_seen_unseen"] = pd.DataFrame(rowsS8)

# ------------------------------------------------------------------ model form (Table S9)
tables["S9A_screening"] = pd.DataFrame(
    [dict(Feature=f, R2=np.corrcoef(qc[f], y_qc)[0, 1] ** 2, r=np.corrcoef(qc[f], y_qc)[0, 1]) for f in FEATURES]
).sort_values("R2", ascending=False)
rowsS9B = []
for d in DESC:
    pg, kg = loo_predict(qc[d].values, y_qc, log=True)
    rowsS9B.append(dict(Descriptor=DESC[d], linear_R2=metrics(y_qc, pred[("QC", d)])["R2"],
                        power_R2=metrics(y_qc, pg)["R2"], linear_MAE=metrics(y_qc, pred[("QC", d)])["MAE"],
                        power_MAE=metrics(y_qc, pg)["MAE"], exponent_mean=kg.mean(), exponent_sd=kg.std()))
tables["S9B_power_law"] = pd.DataFrame(rowsS9B)
X = qc[list(FEATURES)].values; n = len(y_qc); pl = np.empty(n); selected = []
for i in range(n):
    m = np.arange(n) != i
    mod = make_pipeline(StandardScaler(), LassoCV(cv=5, random_state=0, max_iter=50000)).fit(X[m], y_qc[m])
    pl[i] = mod.predict(X[i:i + 1])[0]
    selected += list(np.array(list(FEATURES))[mod[-1].coef_ != 0])
mL, _, rL = boot_stats(y_qc, pl, idx_qc)
tables["S9C_lasso"] = pd.DataFrame([dict(**metrics(y_qc, pl),
                                         R2_CI=np.percentile(rL, [2.5, 97.5]).round(3).tolist(),
                                         MAE_CI=np.percentile(mL, [2.5, 97.5]).round(3).tolist())])
tables["S9C_selection"] = pd.Series(selected).value_counts().rename("folds").to_frame()

# ------------------------------------------------------------------ leave-one-orchard-out (Table S10)
rowsS10 = []
for d in DESC:
    P = pd.Series(index=qc.index, dtype=float); r = dict(Descriptor=DESC[d])
    for date in sorted(qc["date"].unique()):
        tr, te = qc[qc["date"] != date], qc[qc["date"] == date]
        b1, b0 = np.polyfit(tr[d], tr["Weight"], 1); P[te.index] = b1 * te[d] + b0
        mt = metrics(te["Weight"].values, P[te.index].values)
        r[f"test_{date}_R2"], r[f"test_{date}_MAE"], r[f"n_{date}"] = mt["R2"], mt["MAE"], len(te)
    pooled = metrics(y_qc, P.values)
    r["pooled_R2"], r["pooled_MAE"] = pooled["R2"], pooled["MAE"]
    r["p_intercept"] = smf.ols(f"Weight ~ {d} + C(date)", data=qc).fit().pvalues.filter(like="C(date)").iloc[0]
    r["p_slope"] = smf.ols(f"Weight ~ {d} * C(date)", data=qc).fit().pvalues.filter(like=":").iloc[0]
    rowsS10.append(r)
tables["S10_orchard"] = pd.DataFrame(rowsS10)

# ------------------------------------------------------------------ figures
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
UNITS = {"area": ("Projected area (pixel cluster$^{-1}$)", (0, 125000)),
         "voxel": ("Occupied voxel volume (m$^{3}$ cluster$^{-1}$)", (0, 0.004)),
         "hull": ("Convex hull volume (m$^{3}$ cluster$^{-1}$)", (0, 0.05)),
         "dsa": ("Depth-scaled projected area (m$^{2}$ cluster$^{-1}$)", None)}


def fmt_slope(v):
    e = int(np.floor(np.log10(abs(v))))
    return f"{v:.2f}" if e >= -1 else f"{v / 10 ** e:.2f}×10$^{{{e}}}$"


def scatter_panel(ax, x, y, d, letter):
    ax.scatter(x, y, s=55, color="k", zorder=3)
    b1, b0 = np.polyfit(x, y, 1); xs = np.linspace(x.min(), x.max(), 50)
    ax.plot(xs, b1 * xs + b0, color="red", ls=(0, (4, 3)), lw=1.8)
    ax.text(0.72, 0.88, f"y = {fmt_slope(b1)}x + {b0:.4f}\nR² = {np.corrcoef(x, y)[0, 1] ** 2:.3f}",
            transform=ax.transAxes, ha="center", va="center", fontsize=9)
    label, lim = UNITS[d]
    ax.set_xlim(lim if lim else (0, np.ceil(x.max() * 20) / 20)); ax.set_ylim(0, 5)
    ax.set_xlabel(label); ax.set_ylabel("Weight (kg cluster$^{-1}$)")
    ax.text(-0.2, 1.02, letter, transform=ax.transAxes, fontsize=12, fontweight="bold")


def save(fig, name):
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT_DIR, "figures", f"{name}.{ext}"), dpi=300)
    plt.close(fig)


fig, axs = plt.subplots(2, 2, figsize=(9, 8))
for ax, d, L in zip(axs.ravel(), ["area", "voxel", "dsa", "hull"], "ABCD"):
    scatter_panel(ax, noqc[d].values, noqc["Weight"].values, d, L)
save(fig, "Figure5_descriptors_noQC")

fig, axs = plt.subplots(1, 3, figsize=(13.5, 4))
for ax, d, L in zip(axs, ["dsa", "voxel", "hull"], "ABC"):
    scatter_panel(ax, qc[d].values, qc["Weight"].values, d, L)
save(fig, "Figure7_descriptors_QC")

fig, ax = plt.subplots(figsize=(6, 3.8)); cv, cn = curves["voxel"], curves["null"]
ax.fill_between(cv["clusters"], cv["p5"], cv["p95"], color="#0072B2", alpha=0.2, lw=0,
                label="Voxel volume, 5th–95th percentile")
ax.plot(cv["clusters"], cv["mean"], color="#0072B2", lw=1.8, label="Voxel volume, mean")
ax.plot(cn["clusters"], cn["p95"], color="#D55E00", lw=1.3, ls="--",
        label="Null baseline (count × mean weight), 95th percentile")
for t in (20, 10):
    ax.axhline(t, color="grey", ls=":", lw=1)
    for c, col in ((cv, "#0072B2"), (cn, "#D55E00")):
        g = threshold(c, t); ax.plot(g, c.loc[c["clusters"] == g, "p95"].values[0], "o", color=col, ms=5)
ax.set_ylim(0, 80); ax.set_xlim(0, len(y_qc)); ax.grid(alpha=0.3); ax.legend(frameon=False, fontsize=7)
ax.set_xlabel("Number of clusters aggregated"); ax.set_ylabel("Absolute aggregate weight error (%)")
save(fig, "Figure8_aggregation")

fig, ax = plt.subplots(figsize=(4.8, 3.4)); v = tables["S1_views"]
for d, col, mk in [("dsa", "#009E73", "o"), ("voxel", "#0072B2", "s"), ("area", "#E69F00", "^"), ("hull", "#999999", "v")]:
    ax.errorbar(v["views"], v[f"{d}_mean"], yerr=v[f"{d}_sd"], label=DESC[d], color=col, marker=mk, ms=4, capsize=2)
ax.axhline(0, color="k", lw=0.6); ax.grid(alpha=0.3); ax.legend(frameon=False, fontsize=7, loc="center right")
ax.set_xlabel("Number of views averaged per cluster"); ax.set_ylabel("LOOCV R²")
save(fig, "FigureS1_views")

# ------------------------------------------------------------------ save
with pd.ExcelWriter(os.path.join(OUT_DIR, "calibration_results.xlsx")) as xw:
    for name, t in tables.items():
        t.to_excel(xw, sheet_name=name[:31])
print(tables["S7_QC_summary"].to_string(index=False))
print(tables["T5_LOOCV"].round(3).to_string(index=False))
print(tables["F8_aggregation_thresholds"].round(1).to_string(index=False))
print(f"\nSaved: {OUT_DIR}/calibration_results.xlsx and {OUT_DIR}/figures/")
