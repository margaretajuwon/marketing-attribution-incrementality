"""Marketing measurement on SYNTHETIC data: attribution models vs ground truth, a geo-holdout incrementality test,
and a small marketing-mix model. Because the data is simulated, the TRUE incremental effect of every channel is known,
so each method can be graded. Run: python analysis.py -> results/*.png, results/*.csv, README.md
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

OUT = Path(__file__).parent; RES = OUT / "results"; RES.mkdir(exist_ok=True)
rng = np.random.default_rng(21)
CH = ["search", "social", "email", "display"]

# ============================================================ A. journeys and attribution
N = 40_000
intent = rng.normal(0, 1, N)                                   # latent purchase intent (the confounder)
effect = {"search": 0.80, "social": 0.55, "email": 0.35, "display": 0.03}   # TRUE causal effect on log-odds
# exposure depends on intent: high-intent users are more likely to be searched/retargeted (selection bias)
p_touch = {"search": 1 / (1 + np.exp(-(-0.6 + 0.9 * intent))), "display": 1 / (1 + np.exp(-(-0.2 + 0.8 * intent))),
           "social": np.full(N, 0.30), "email": np.full(N, 0.35)}
T = {c: (rng.random(N) < p_touch[c]).astype(int) for c in CH}
base = -3.6 + 0.9 * intent
def p_conv(drop=None):
    z = base + sum(effect[c] * T[c] for c in CH if c != drop)
    return 1 / (1 + np.exp(-z))
p_full = p_conv()
conv = (rng.random(N) < p_full).astype(int)
truth = {c: float((p_full - p_conv(drop=c)).sum()) for c in CH}     # true incremental conversions per channel
truth_share = pd.Series(truth) / sum(truth.values())

# order of touches per user: social/email/search early-to-mid, display (retargeting) tends to come last
rank = {"social": 0.0, "email": 0.5, "search": 1.0, "display": 2.0}
paths = []
for i in range(N):
    ts = [c for c in CH if T[c][i]]
    ts.sort(key=lambda c: rank[c] + rng.normal(0, 0.6))
    paths.append(ts)
conv_idx = np.where(conv == 1)[0]

def credit_last():
    s = pd.Series(0.0, index=CH)
    for i in conv_idx:
        if paths[i]: s[paths[i][-1]] += 1
    return s
def credit_first():
    s = pd.Series(0.0, index=CH)
    for i in conv_idx:
        if paths[i]: s[paths[i][0]] += 1
    return s
def credit_linear():
    s = pd.Series(0.0, index=CH)
    for i in conv_idx:
        for c in paths[i]: s[c] += 1 / len(paths[i])
    return s
def credit_markov():
    states = ["start"] + CH + ["conv", "null"]
    idx = {s: k for k, s in enumerate(states)}
    M = np.zeros((len(states), len(states)))
    for i, pth in enumerate(paths):
        seq = ["start"] + pth + (["conv"] if conv[i] else ["null"])
        for a, b in zip(seq[:-1], seq[1:]): M[idx[a], idx[b]] += 1
    M = M / M.sum(axis=1, keepdims=True).clip(min=1)
    def p_conv_chain(mat):
        tr = [idx[s] for s in ["start"] + CH]
        Q = mat[np.ix_(tr, tr)]; R = mat[tr, idx["conv"]]
        return np.linalg.solve(np.eye(len(tr)) - Q, R)[0]
    full = p_conv_chain(M); out = {}
    for c in CH:
        m2 = M.copy(); k = idx[c]; m2[:, k] = 0; m2[k, :] = 0
        m2[:, idx["null"]] += (M[:, k]); 
        out[c] = full - p_conv_chain(m2)
    s = pd.Series(out).clip(lower=0)
    return s / s.sum() * conv.sum()

att = pd.DataFrame({"last_click": credit_last(), "first_click": credit_first(), "linear": credit_linear(), "markov_removal": credit_markov()})
share = att / att.sum()
share["TRUTH_incremental"] = truth_share
err = (share.drop(columns="TRUTH_incremental").sub(share["TRUTH_incremental"], axis=0)).abs().sum() / 2 * 100   # total share misallocated (pp)
share.round(3).to_csv(RES / "attribution_share.csv")
ax = (share * 100).plot.bar(figsize=(8, 4), rot=0, width=0.8)
ax.set_ylabel("% of conversions credited"); ax.set_title("Attribution models vs true incremental contribution")
plt.tight_layout(); plt.savefig(RES / "attribution_vs_truth.png", dpi=140); plt.close()

# ============================================================ B. geo holdout test (difference-in-differences)
G, PRE, TEST = 60, 12, 4
geo_base = rng.lognormal(6, 0.5, G)
season = 1 + 0.15 * np.sin(np.linspace(0, 3, PRE + TEST))           # shared trend/seasonality
treat = np.zeros(G, bool); treat[rng.choice(G, G // 2, replace=False)] = True
TRUE_LIFT = 0.04                                                   # +4% conversions in treated geos during the test
wk = np.arange(PRE + TEST)
Y = rng.poisson(np.outer(geo_base, season) * np.where((wk >= PRE)[None, :] & treat[:, None], 1 + TRUE_LIFT, 1))
pre = Y[:, :PRE].mean(1); post = Y[:, PRE:].mean(1)
naive = post[treat].mean() / pre[treat].mean() - 1                          # pre/post on treated only: seasonality contaminates it
growth = np.log(post / pre)
did = growth[treat].mean() - growth[~treat].mean()
perm = []
for _ in range(5000):
    t2 = rng.permutation(treat); perm.append(growth[t2].mean() - growth[~t2].mean())
perm = np.array(perm); ci = np.percentile(perm, [2.5, 97.5])
p_val = float((np.abs(perm) >= abs(did)).mean())

# ============================================================ C. small marketing-mix model
W = 104
spend = pd.DataFrame({c: rng.gamma(4, m, W) * (1 + 0.3 * np.sin(np.arange(W) / 8 + k)) for k, (c, m) in enumerate(zip(CH, [30, 22, 6, 18]))})
def adstock(x, d):
    o = np.zeros_like(x); 
    for t in range(len(x)): o[t] = x[t] + (d * o[t - 1] if t else 0)
    return o
decay = {"search": 0.2, "social": 0.5, "email": 0.1, "display": 0.6}
sat = lambda x, h: x / (x + h)
true_beta = {"search": 900, "social": 700, "email": 250, "display": 40}
ad = {c: sat(adstock(spend[c].to_numpy(), decay[c]), 80) for c in CH}
trend = 1500 + np.arange(W) * 3 + 120 * np.sin(np.arange(W) * 2 * np.pi / 52)
y = trend + sum(true_beta[c] * ad[c] for c in CH) + rng.normal(0, 40, W)
X = np.column_stack([ad[c] for c in CH] + [np.arange(W), np.sin(np.arange(W) * 2 * np.pi / 52), np.cos(np.arange(W) * 2 * np.pi / 52)])
mdl = Ridge(alpha=1e-3).fit(X, y); est = dict(zip(CH, mdl.coef_[:4]))
boots = []
for _ in range(300):
    b = rng.choice(W, W); boots.append(Ridge(alpha=1e-3).fit(X[b], y[b]).coef_[:4])
boots = np.array(boots); lo, hi = np.percentile(boots, [5, 95], axis=0)
mmm = pd.DataFrame({"true_effect": true_beta, "estimated": est, "ci90_low": dict(zip(CH, lo)), "ci90_high": dict(zip(CH, hi))}).round(0)
mmm.to_csv(RES / "mmm_effects.csv")
r2 = mdl.score(X, y)

# ============================================================ README
def tbl(df, fmt="{:.1f}"):
    h = "| " + " | ".join([df.index.name or ""] + list(df.columns)) + " |\n|" + "---|" * (len(df.columns) + 1) + "\n"
    return h + "\n".join("| " + " | ".join([str(i)] + [fmt.format(v) for v in r]) + " |" for i, r in df.iterrows())
sh = (share * 100).round(1); sh.index.name = "channel"
readme = f"""# Marketing attribution and incrementality

**Synthetic data, real method.** Everything is simulated with fixed seeds in `analysis.py`; no employer data. Because the simulation
knows each channel's *true* causal effect, every method can be graded instead of just trusted.

## Questions
1. Do last-click, first-click, linear and data-driven (Markov removal effect) attribution tell us where conversions really come from?
2. Does a geo-holdout test recover a known lift when a simple before/after comparison cannot?
3. Can a small marketing-mix model recover channel effects from weekly spend alone?

## 1. Attribution vs ground truth (40,000 users, intent confounds exposure)
High-intent users are more likely to be searched and retargeted, so rules that reward the final touch over-credit those channels.
Display has almost no true effect but is mostly shown last (retargeting).

Share of conversions credited, %:

{tbl(sh)}

Total share misallocated versus truth (percentage points): last-click **{err['last_click']:.1f}**, first-click **{err['first_click']:.1f}**,
linear **{err['linear']:.1f}**, Markov removal **{err['markov_removal']:.1f}**.
Takeaway: every path-based rule misallocates credit, and which one looks best depends on assumptions about touch order, so none of them can be trusted for budget decisions. Last-click is the worst here because it hands most credit to display retargeting, which has almost no true effect. Use them for reporting and calibrate budget decisions with experiments (section 2).

![attribution](results/attribution_vs_truth.png)

## 2. Geo-holdout incrementality test (60 geos, 12 pre weeks + 4 test weeks, half randomized to treatment)
True lift built into the simulation: **+{TRUE_LIFT*100:.0f}%**.
- Naive before/after on treated geos: **{naive*100:+.1f}%** (contaminated by the shared seasonal trend).
- Difference-in-differences vs holdout geos: **{(np.exp(did)-1)*100:+.1f}%**, permutation 95% interval [{(np.exp(ci[0])-1)*100:+.1f}%, {(np.exp(ci[1])-1)*100:+.1f}%], p = {p_val:.3f}.
Randomizing geos and comparing against a holdout removes the trend that a before/after comparison cannot.

## 3. Marketing-mix model (104 weeks, adstock + saturation, ridge regression)
In-sample R^2 {r2:.2f}. True vs estimated channel effects (conversions at saturation), with 90% bootstrap intervals:

{tbl(mmm.rename_axis('channel'), '{:.0f}')}

## Caveats I would raise on a real engagement
- Attribution and MMM need to be calibrated with experiments (like section 2); neither is a causal estimate on its own.
- The MMM here is given the true adstock and saturation form. On real data those must be estimated, and weekly data with correlated
  spend gives much wider intervals.
- A holdout test measures one channel at one spend level; extrapolating beyond it needs more cells.

## Run it
`pip install numpy pandas scikit-learn matplotlib` then `python analysis.py`.
"""
(OUT / "README.md").write_text(readme, encoding="utf-8")
print(sh.to_string()); print(err.round(1).to_dict())
print("geo: naive", round(naive*100,2), "DiD", round((np.exp(did)-1)*100,2), "CI", (np.exp(ci)-1).round(3), "p", p_val)
print(mmm.to_string(), r2)

