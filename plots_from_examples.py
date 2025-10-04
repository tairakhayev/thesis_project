# plots_from_examples.py
import os, numpy as np, matplotlib.pyplot as plt
from scipy.signal import welch, coherence

EX_DIR = os.path.join("final")
OUT    = os.path.join("final","plots")
os.makedirs(OUT, exist_ok=True)

def shade_band(ax, f1, f2, label=None):
    ax.axvspan(f1, f2, alpha=0.15, label=label if label else None)

def plot_psd_true_vs_pred(ex_path, ch="O1"):
    dat = np.load(ex_path, allow_pickle=True)
    fs = int(dat["fs"])
    chs = [str(c) for c in dat["channels"]]
    H   = dat["true_future"][0].shape[1]

    if ch not in chs:
        print(f"[skip PSD] {ch} not in {chs}")
        return
    idx = chs.index(ch)

    y_true = dat["true_future"][0][idx]  # (H,)
    y_pred = dat["pred_future"][0][idx]

    f_t, p_t = welch(y_true, fs=fs, nperseg=min(128, H))
    f_p, p_p = welch(y_pred, fs=fs, nperseg=min(128, H))

    plt.figure(figsize=(6,4))
    plt.semilogy(f_t, p_t, label="True")
    plt.semilogy(f_p, p_p, label="Predicted", linestyle="--")

    # bands
    shade_band(plt.gca(), 1, 4,  "delta")
    shade_band(plt.gca(), 4, 8,  "theta")
    shade_band(plt.gca(), 8, 13, "alpha")
    shade_band(plt.gca(), 13,30, "beta")
    shade_band(plt.gca(), 30,45, "low-γ")

    plt.xlim(0, 45)
    plt.xlabel("Hz"); plt.ylabel("PSD")
    plt.title(f"Welch PSD — True vs Pred ({ch})")
    plt.legend(loc="upper right")
    plt.tight_layout()

    base = os.path.splitext(os.path.basename(ex_path))[0]
    plt.savefig(os.path.join(OUT, f"{base}_psd_{ch}.png")); plt.close()

def plot_alpha_coherence_true_vs_pred(ex_path, a="O1", b="O2"):
    dat = np.load(ex_path, allow_pickle=True)
    fs = int(dat["fs"])
    chs = [str(c) for c in dat["channels"]]

    if a not in chs or b not in chs:
        print(f"[skip COH] missing pair {a}-{b} in {chs}")
        return
    ia, ib = chs.index(a), chs.index(b)

    ta = dat["true_future"][0][ia]; tb = dat["true_future"][0][ib]
    pa = dat["pred_future"][0][ia]; pb = dat["pred_future"][0][ib]

    f, coh_true = coherence(ta, tb, fs=fs, nperseg=min(128, len(ta)))
    _, coh_pred = coherence(pa, pb, fs=fs, nperseg=min(128, len(pa)))

    plt.figure(figsize=(6,4))
    plt.plot(f, coh_true, label="True")
    plt.plot(f, coh_pred, label="Predicted", linestyle="--")
    plt.axvspan(8, 13, alpha=0.15, label="alpha")
    plt.xlim(0, 45); plt.ylim(0, 1.0)
    plt.xlabel("Hz"); plt.ylabel("Coherence")
    plt.title(f"Coherence — True vs Pred ({a}-{b})")
    plt.legend()
    plt.tight_layout()

    base = os.path.splitext(os.path.basename(ex_path))[0]
    plt.savefig(os.path.join(OUT, f"{base}_coh_{a}{b}.png")); plt.close()

# run on all saved examples
for f in sorted(os.listdir(EX_DIR)):
    if f.endswith(".npz"):
        ex_path = os.path.join(EX_DIR, f)
        # pick two representative channels/pairs
        for ch in ("Cz", "O1"):
            plot_psd_true_vs_pred(ex_path, ch=ch)
        plot_alpha_coherence_true_vs_pred(ex_path, a="O1", b="O2")

print("Saved spectral/coherence plots to:", OUT)
