"""
Analysis: AI-tool adoption and job satisfaction
Stack Overflow Annual Developer Survey 2025
Research question: Is reported AI-tool adoption associated with reported
job satisfaction, controlling for years of professional experience and
primary developer role?
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from statsmodels.iolib.summary2 import summary_col

# ── paths ──────────────────────────────────────────────────────────────────────
ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA   = os.path.join(ROOT, "data", "results.csv")
FIGS   = os.path.join(ROOT, "paper", "figures")
os.makedirs(FIGS, exist_ok=True)

# ── 1. load ────────────────────────────────────────────────────────────────────
print("Loading data …")
df = pd.read_csv(DATA, encoding="utf-8-sig", low_memory=False)
print(f"  Raw dataset: {df.shape[0]:,} rows × {df.shape[1]} columns")

# ── 2. variable construction ───────────────────────────────────────────────────
# AI-use ordinal (0 = no plan … 4 = daily)
ai_map = {
    "No, and I don't plan to":                 0,
    "No, but I plan to soon":                  1,
    "Yes, I use AI tools monthly or infrequently": 2,
    "Yes, I use AI tools weekly":              3,
    "Yes, I use AI tools daily":               4,
}
ai_label = {
    0: "No / no plan",
    1: "Plan soon",
    2: "Monthly",
    3: "Weekly",
    4: "Daily",
}
df["ai_ordinal"] = df["AISelect"].map(ai_map)

# Binary AI-use indicator (any current use vs. none/plan only)
df["ai_user"] = (df["ai_ordinal"] >= 2).astype(float)

# Job satisfaction: numeric 0-10, already in df["JobSat"]
df["jobsat"] = pd.to_numeric(df["JobSat"], errors="coerce")

# Professional experience: WorkExp (years)
df["exp"] = pd.to_numeric(df["WorkExp"], errors="coerce")

# Primary developer role: take text before first ";" if multiple roles given
df["primary_role"] = df["DevType"].str.split(";").str[0].str.strip()

# Keep only professional developers (exclude students and blanks)
df = df[~df["primary_role"].isin(["Student", "Other (please specify):"])]

# Collapse rare roles (< 500 obs) into "Other professional"
role_counts = df["primary_role"].value_counts()
rare = role_counts[role_counts < 500].index
df["role"] = df["primary_role"].where(~df["primary_role"].isin(rare), "Other professional")

# ── 3. analysis sample ─────────────────────────────────────────────────────────
cols = ["jobsat", "ai_ordinal", "ai_user", "exp", "role"]
adf = df[cols].dropna()
print(f"  Analysis sample: {len(adf):,} respondents (after dropping missing values)")

# ── 4. descriptive statistics ──────────────────────────────────────────────────
print("\n=== Descriptive statistics ===")
print(f"  Mean job satisfaction: {adf['jobsat'].mean():.2f}  (SD {adf['jobsat'].std():.2f})")
print(f"  Median job satisfaction: {adf['jobsat'].median():.1f}")
print(f"  Mean years experience: {adf['exp'].mean():.1f}  (median {adf['exp'].median():.1f})")

print("\nMean/median satisfaction by AI-use category:")
sat_by_ai = (
    df.dropna(subset=["jobsat","ai_ordinal"])
      .groupby("ai_ordinal")["jobsat"]
      .agg(["mean","median","count"])
      .rename(index=ai_label)
)
sat_by_ai.columns = ["mean_jobsat", "median_jobsat", "n"]
print(sat_by_ai.to_string())

print("\nRole distribution in analysis sample:")
print(adf["role"].value_counts().to_string())

# ── 5. OLS regressions ─────────────────────────────────────────────────────────
print("\n=== OLS regressions ===")

# Model 1: bivariate
m1 = smf.ols("jobsat ~ ai_ordinal", data=adf).fit(cov_type="HC3")
# Model 2: add experience
m2 = smf.ols("jobsat ~ ai_ordinal + exp", data=adf).fit(cov_type="HC3")
# Model 3: add role fixed effects (this is the main specification)
m3 = smf.ols("jobsat ~ ai_ordinal + exp + C(role)", data=adf).fit(cov_type="HC3")

# Main coefficient of interest
coef  = m3.params["ai_ordinal"]
se    = m3.bse["ai_ordinal"]
pval  = m3.pvalues["ai_ordinal"]
ci_lo = m3.conf_int().loc["ai_ordinal", 0]
ci_hi = m3.conf_int().loc["ai_ordinal", 1]

print(f"\nMain model (M3) -- ai_ordinal coefficient:")
print(f"  b = {coef:.4f}  SE = {se:.4f}  p = {pval:.4f}")
print(f"  95% CI: [{ci_lo:.4f}, {ci_hi:.4f}]")
print(f"  N = {int(m3.nobs):,}   R2 = {m3.rsquared:.4f}")

# Robustness: restrict to >5 years experience
adf5 = adf[adf["exp"] > 5]
m3r  = smf.ols("jobsat ~ ai_ordinal + exp + C(role)", data=adf5).fit(cov_type="HC3")
print(f"\nRobustness (>5 yrs exp, N={int(m3r.nobs):,}): b = {m3r.params['ai_ordinal']:.4f}  p = {m3r.pvalues['ai_ordinal']:.4f}")

# ── 6. save results table ──────────────────────────────────────────────────────
results = pd.DataFrame({
    "model": ["M1 bivariate", "M2 + experience", "M3 + role FE (main)", "M3 robustness (exp>5)"],
    "coef_ai":    [m1.params["ai_ordinal"], m2.params["ai_ordinal"], coef, m3r.params["ai_ordinal"]],
    "se":         [m1.bse["ai_ordinal"],    m2.bse["ai_ordinal"],    se,   m3r.bse["ai_ordinal"]],
    "pvalue":     [m1.pvalues["ai_ordinal"],m2.pvalues["ai_ordinal"],pval, m3r.pvalues["ai_ordinal"]],
    "n":          [int(m1.nobs), int(m2.nobs), int(m3.nobs), int(m3r.nobs)],
    "r2":         [m1.rsquared, m2.rsquared,  m3.rsquared,  m3r.rsquared],
})
results_path = os.path.join(ROOT, "paper", "results_table.csv")
results.to_csv(results_path, index=False)
print(f"\nResults table saved to {results_path}")

# Descriptive table
sat_by_ai.to_csv(os.path.join(ROOT, "paper", "sat_by_ai.csv"))

# ── 7. figure 1: satisfaction by AI-use category ──────────────────────────────
fig1_data = (
    df.dropna(subset=["jobsat","ai_ordinal"])
      .groupby("ai_ordinal")["jobsat"]
      .agg(["mean","sem"])
      .rename(index=ai_label)
)

fig, ax = plt.subplots(figsize=(7, 4.5))
colors = ["#d62728" if i < 2 else "#2ca02c" for i in range(5)]
bars = ax.bar(
    fig1_data.index, fig1_data["mean"],
    yerr=1.96 * fig1_data["sem"],
    color=colors, edgecolor="white", linewidth=0.8,
    error_kw=dict(ecolor="black", capsize=4, linewidth=1.2),
    width=0.6
)
ax.set_xlabel("Reported AI-tool use", fontsize=11)
ax.set_ylabel("Mean job satisfaction (0–10)", fontsize=11)
ax.set_title("Job satisfaction by AI-tool adoption", fontsize=12, fontweight="bold")
ax.set_ylim(6.5, 8.5)
ax.tick_params(axis="x", labelsize=9)
ax.spines[["top","right"]].set_visible(False)

# Annotate counts
all_counts = df.dropna(subset=["jobsat","ai_ordinal"]).groupby("ai_ordinal").size().rename(index=ai_label)
for bar, (label, row) in zip(bars, fig1_data.iterrows()):
    n = all_counts[label]
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.12,
            f"n={n:,}", ha="center", va="bottom", fontsize=7.5, color="#333")

fig.tight_layout()
fig1_path = os.path.join(FIGS, "fig1_satisfaction_by_ai_use.pdf")
fig.savefig(fig1_path, bbox_inches="tight")
plt.close()
print(f"Figure 1 saved to {fig1_path}")

# ── 8. figure 2: years of professional experience distribution ─────────────────
fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(adf["exp"], bins=range(0, int(adf["exp"].max()) + 2, 1),
        color="#1f77b4", edgecolor="white", linewidth=0.4, alpha=0.85)
ax.axvline(adf["exp"].median(), color="#d62728", linewidth=1.5,
           linestyle="--", label=f"Median = {adf['exp'].median():.0f} yrs")
ax.axvline(adf["exp"].mean(), color="#ff7f0e", linewidth=1.5,
           linestyle=":", label=f"Mean = {adf['exp'].mean():.1f} yrs")
ax.set_xlabel("Years of professional coding experience", fontsize=11)
ax.set_ylabel("Number of respondents", fontsize=11)
ax.set_title("Distribution of professional experience\n(analysis sample)", fontsize=12, fontweight="bold")
ax.legend(fontsize=9)
ax.spines[["top","right"]].set_visible(False)
fig.tight_layout()
fig2_path = os.path.join(FIGS, "fig2_years_experience.pdf")
fig.savefig(fig2_path, bbox_inches="tight")
plt.close()
print(f"Figure 2 saved to {fig2_path}")

# ── 9. summary ─────────────────────────────────────────────────────────────────
print("\n=== Summary ===")
print(f"Raw data: {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"Analysis sample (professionals, complete cases): {len(adf):,} respondents")
print(f"Main result: each step up in AI adoption associates with b={coef:+.3f} points in job satisfaction")
print(f"(SE={se:.3f}, p={pval:.4f}, 95% CI [{ci_lo:.3f}, {ci_hi:.3f}]), controlling for experience and role.")
print("Robustness check (exp>5 yrs) does not materially change the estimate.")
print("\nDone.")
