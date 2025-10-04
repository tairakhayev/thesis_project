# run_forecast_pbi_windows.py
import os, glob, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import mne
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

from forecast_models import set_seed, ResidualRidgeHankelForecaster
from pbi import pbi_for_window
warnings.filterwarnings("ignore", message="invalid value encountered")

SEED=42; set_seed(SEED)

# --- конфиг ---
DATA_DIR = "/Users/tairakhayev/Desktop/forecasting/dataverse_files"
OUT_DIR  = os.getcwd()
FS = 250
BP = (1.0, 40.0)
NOTCH = [50.0]
CHANNELS = ["Fp1","Fp2","F3","Fz","F4","C3","Cz","C4","P3","Pz","P4","O1","O2"]

L_sec, H_sec = 3.0, 0.5
L, H = int(L_sec*FS), int(H_sec*FS)

STRIDE_MULT = 2
MAX_WINDOWS_PER_SUBJ = 400
SPIKE_Z_THR, FLAT_STD_THR, OUT_RMS_THR = 8.0, 1e-3, 0.03
AR_M = 150

# визуализация
PLOT_DIR = os.path.join("final", "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

# --- helpers ---
def is_healthy(path_or_name):
    return Path(path_or_name).stem.lower().startswith('h')

def lowpass_filter(x, fs, cutoff=25.0, order=4):
    """x: (C,T) -> (C,T), по каналам"""
    if x.shape[1] < (order*3 + 1):
        return x  # слишком коротко для filtfilt — оставим как есть
    b, a = butter(order, cutoff/(0.5*fs), btype="low")
    return filtfilt(b, a, x, axis=1)

def smooth_future(y, k=5):
    """скользящее среднее по времени: y (C,H)"""
    if k <= 1 or k > y.shape[1]:
        return y
    pad = k//2
    ypad = np.pad(y, ((0,0),(pad,pad)), mode='edge')
    ker = np.ones(k, dtype=np.float32)/k
    sm = np.apply_along_axis(lambda v: np.convolve(v, ker, mode='valid'), axis=1, arr=ypad)
    return sm.astype(y.dtype)

def load_one_edf(path, keep_channels):
    raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
    picks = [ch for ch in keep_channels if ch in raw.ch_names]
    raw.pick(picks)
    raw.notch_filter(NOTCH, fir_design='firwin', verbose=False)
    raw.filter(BP[0], BP[1], fir_design='firwin', verbose=False)
    if int(raw.info['sfreq']) != FS:
        raw.resample(FS)
    X = raw.get_data()
    ch_std_uv = X.std(axis=1, keepdims=True) + 1e-8
    Xz = (X - X.mean(axis=1, keepdims=True)) / ch_std_uv
    return Xz.astype(np.float32), [ch for ch in keep_channels if ch in raw.ch_names]

def window_bad(Xw):
    if np.max(np.abs(Xw)) > SPIKE_Z_THR: return True
    if np.any(np.std(Xw, axis=1) < FLAT_STD_THR): return True
    return False

def make_LH_windows(X, L, H, stride_mult=1, max_windows=600, fs=FS, lp_cut=25.0):
    """Дополнительно low-pass и к истории, и к таргету"""
    C, T = X.shape
    step = max(int(L * stride_mult), 1)
    Xw, Yw = [], []
    for t0 in range(0, max(T - (L + H) + 1, 1), step):
        xin  = X[:, t0:t0+L]
        yout = X[:, t0+L:t0+L+H]
        if xin.shape[1] < L or yout.shape[1] < H: break
        if np.sqrt(np.mean(yout**2)) < OUT_RMS_THR: continue
        if window_bad(xin) or window_bad(yout):     continue

        # доп. low-pass и к истории, и к будущему
        xin_f  = lowpass_filter(xin,  fs, cutoff=lp_cut)
        yout_f = lowpass_filter(yout, fs, cutoff=lp_cut)

        Xw.append(xin_f); Yw.append(yout_f)
        if len(Xw) >= max_windows: break

    if not Xw:
        return np.zeros((1,C,L), np.float32), np.zeros((1,C,H), np.float32)
    return np.stack(Xw, axis=0), np.stack(Yw, axis=0)

def fit_ar2_from_tail(x, M=150):
    M = min(M, len(x)-2)
    if M < 10: return 1.0, 0.0
    y = x[-M:]; Y = y[2:]; Phi = np.stack([y[1:-1], y[:-2]], axis=1)
    lam = 1e-3; A = Phi.T @ Phi + lam*np.eye(2); b = Phi.T @ Y
    a = np.linalg.solve(A, b); return float(a[0]), float(a[1])

def ar2_forecast(history, H, M=150):
    C, _ = history.shape
    out = np.zeros((C,H), dtype=history.dtype)
    for c in range(C):
        x = history[c]; a1, a2 = fit_ar2_from_tail(x, M)
        y1, y2 = x[-1], x[-2]
        for t in range(H):
            y = a1*y1 + a2*y2; out[c,t] = y; y2, y1 = y1, y
    return out

def choose_lambda_delta(X_tr, D_tr, seed=SEED):
    n = len(X_tr); idx = np.arange(n)
    rng = np.random.RandomState(seed); rng.shuffle(idx)
    cut = max(512, int(0.2*n)); va, tr = idx[:cut], idx[cut:]
    Xtr, Dtr = X_tr[tr], D_tr[tr]; Xva, Dva = X_tr[va], D_tr[va]
    best, best_lam = -1e9, 1e-3
    # более сильная регуляризация
    for lam in [1e-3, 3e-3, 1e-2, 3e-2, 1e-1]:
        m = ResidualRidgeHankelForecaster(lam=lam, norm_by_hist_std=True).fit(Xtr, Dtr)
        P = m.predict(Xva)
        rm = np.sqrt(np.mean((Dva - P)**2))
        r0 = np.sqrt(np.mean((Dva - 0.0)**2)) + 1e-9
        score = (r0 - rm) / (np.sqrt(np.mean(Dva**2)) + 0.1)
        if score > best: best, best_lam = score, lam
    return best_lam

def plot_example(Xte, Yte, base_te, Yhat, chs, subject, win_idx, max_ch=3):
    plt.figure(figsize=(12, 6))
    for c, ch in enumerate(chs[:max_ch]):
        t_hist = np.arange(L) / FS
        t_future = np.arange(L, L+H) / FS
        plt.subplot(max_ch, 1, c+1)
        plt.plot(t_hist, Xte[win_idx, c], color="black", label="History" if c==0 else "")
        plt.plot(t_future, Yte[win_idx, c], color="green", label="True Future" if c==0 else "")
        plt.plot(t_future, base_te[win_idx, c], color="red", linestyle="--", label="AR(2)" if c==0 else "")
        plt.plot(t_future, Yhat[win_idx, c], color="blue", linestyle="--", label="RRHF" if c==0 else "")
        if c==0: plt.legend(loc="upper right")
        plt.ylabel(ch)
    plt.xlabel("Time (sec)")
    plt.suptitle(f"{subject} | Window {win_idx}")
    fname = f"{subject}_win{win_idx}.png"
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, fname))
    plt.close()

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    edfs = sorted(glob.glob(os.path.join(DATA_DIR, "*.edf"))); assert edfs, "Put *.edf to DATA_DIR"

    records, rows = [], []

    # загрузка и нарезка
    for p in edfs:
        Xz, picks = load_one_edf(p, CHANNELS)
        name  = Path(p).stem
        label = 0 if is_healthy(p) else 1
        Xw, Yw = make_LH_windows(Xz, L, H, STRIDE_MULT, MAX_WINDOWS_PER_SUBJ, fs=FS, lp_cut=25.0)
        records.append({"name": name, "label": label, "chs": picks, "Xw": Xw, "Yw": Yw})

    # LOSO
    for i, test in enumerate(records):
        chs_test = test["chs"]
        X_list, D_list = [], []
        for j, r in enumerate(records):
            if j==i: continue
            if r["label"]==1 and (hash((i,j)) % 5) != 0: continue  # ~20% больных
            if not all(c in r["chs"] for c in chs_test): continue
            idx = [r["chs"].index(c) for c in chs_test]
            Xw, Yw = r["Xw"][:, idx, :], r["Yw"][:, idx, :]
            base = np.stack([ar2_forecast(Xw[k], H, M=AR_M) for k in range(Xw.shape[0])], axis=0)
            D = Yw - base
            X_list.append(Xw); D_list.append(D)

        rrhf = None
        if X_list:
            X_tr_all = np.concatenate(X_list, axis=0)
            D_tr_all = np.concatenate(D_list, axis=0)
            lam = choose_lambda_delta(X_tr_all, D_tr_all, seed=SEED+i)
            rrhf = ResidualRidgeHankelForecaster(lam=lam, norm_by_hist_std=True).fit(X_tr_all, D_tr_all)

        Xte, Yte = test["Xw"], test["Yw"]
        if rrhf is None: continue

        base_te = np.stack([ar2_forecast(Xte[k], H, M=AR_M) for k in range(Xte.shape[0])], axis=0)
        Dh = rrhf.predict(Xte)
        Yhat = base_te + Dh

        # мягкий клип + сглаживание прогноза
        last = Xte[:, :, -1:]; hstd = np.std(Xte, axis=2, keepdims=True) + 1e-6
        Yhat = np.clip(Yhat, last - 3.0*hstd, last + 3.0*hstd)
        Yhat = np.stack([smooth_future(Yhat[n], k=5) for n in range(Yhat.shape[0])], axis=0)

        # сохранить несколько графиков (первые 3 субъекта, по 2 окна)
        if i < 3:
            for k in range(min(2, Yte.shape[0])):
                plot_example(Xte, Yte, base_te, Yhat, chs_test, test["name"], k)

        # PBI-фичи
        for k in range(Yte.shape[0]):
            true_seg = Yte[k]; pred_seg = Yhat[k]
            res = true_seg - pred_seg; x_last = Xte[k, :, -1][:, None]
            feats = pbi_for_window(true_seg, pred_seg, res, FS, chs_test, x_last=x_last)
            feats["subject"] = test["name"]; feats["true"] = int(test["label"]); feats["win_idx"] = int(k)
            rows.append(feats)

    df = pd.DataFrame(rows)
    out_csv = os.path.join(OUT_DIR, "pbi_windows.csv")
    df.to_csv(out_csv, index=False)
    print("Saved ->", out_csv)
    print(f"Graphs saved to {PLOT_DIR}")

if __name__ == "__main__":
    main()
