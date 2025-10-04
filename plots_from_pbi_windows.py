# plots_from_pbi_windows.py
import os, pandas as pd, numpy as np, matplotlib.pyplot as plt

CSV = "pbi_windows.csv"
OUT = os.path.join("final","plots")
os.makedirs(OUT, exist_ok=True)

df = pd.read_csv(CSV)
# expected columns: 'true' (0=healthy,1=sick), 'nRMSE', 'delta_nRMSE', 'improve_rate'

plt.figure(figsize=(6,4))
plt.hist(df.loc[df.true==0, "nRMSE"], bins=25, alpha=0.6, density=True, label="Healthy")
plt.hist(df.loc[df.true==1, "nRMSE"], bins=25, alpha=0.6, density=True, label="Sick")
plt.xlabel("nRMSE"); plt.ylabel("density"); plt.title("Forecast error distribution by group")
plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(OUT, "pbi_nrmse_hist_by_group.png")); plt.close()

# Figure A2 — ΔnRMSE vs. Naive (higher = model beats naive more)
plt.figure(figsize=(6,4))
data = [df.loc[df.true==0, "delta_nRMSE"].dropna(), df.loc[df.true==1, "delta_nRMSE"].dropna()]
plt.boxplot(data, labels=["Healthy","Sick"], showmeans=True)
plt.ylabel("ΔnRMSE (Naive − Model)")
plt.title("Relative improvement over naive baseline")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "pbi_delta_nrmse_box.png")); plt.close()

# Figure A3 — Improvement rate (fraction of points where model < naive)
plt.figure(figsize=(6,4))
data = [df.loc[df.true==0, "improve_rate"].dropna(), df.loc[df.true==1, "improve_rate"].dropna()]
plt.boxplot(data, labels=["Healthy","Sick"], showmeans=True)
plt.ylabel("Improvement rate")
plt.title("Pointwise improvement over naive")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "pbi_improve_rate_box.png")); plt.close()

print("Saved:", OUT)
