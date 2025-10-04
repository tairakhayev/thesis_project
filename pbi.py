import numpy as np
from scipy.signal import welch, coherence

# --- robust IQR ---
try:
    from scipy.stats import iqr as _scipy_iqr
    def IQR(x):
        return float(_scipy_iqr(x, nan_policy="omit"))
except Exception:
    def IQR(x):
        x = np.asarray(x)
        q75, q25 = np.nanpercentile(x, 75), np.nanpercentile(x, 25)
        return float(q75 - q25)

BANDS = {"delta":(1,4), "theta":(4,8), "alpha":(8,13), "beta":(13,30), "lowg":(30,45)}
COH_PAIRS = [("C3","C4"),("P3","P4"),("O1","O2")]

def ensure_finite(x): return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

def rmse(a,b): return float(np.sqrt(np.mean((a-b)**2)))
def mae(a,b):  return float(np.mean(np.abs(a-b)))

def lag_rmse(y, yhat, lags):
    out={}
    for L in lags:
        if L<=0 or L>y.shape[-1]: continue
        out[L]=rmse(y[...,L-1], yhat[...,L-1])
    return out

def spectral_nrmse(y, yhat, fs, bands=BANDS):
    C = y.shape[0]
    errs = {k: [] for k in bands}
    for c in range(C):
        f, p_true = welch(y[c], fs=fs, nperseg=min(128, y.shape[1]))
        _, p_pred = welch(yhat[c], fs=fs, nperseg=min(128, y.shape[1]))
        p_true = ensure_finite(p_true); p_pred = ensure_finite(p_pred)
        for name,(f1,f2) in bands.items():
            m = (f>=f1)&(f<=f2)
            t = np.trapz(p_true[m], f[m]); p = np.trapz(p_pred[m], f[m])
            denom = max(1e-6, 0.5*(t + p))
            errs[name].append(abs(t - p)/denom)
    return {k: float(np.mean(v)) if len(v)>0 else 0.0 for k,v in errs.items()}

def autocorr_lag1(x):
    x = x - x.mean()
    v = x.var() + 1e-12
    return float(np.correlate(x[:-1], x[1:])[0] / ((len(x)-1)*v))

def line_length(x): return float(np.mean(np.abs(np.diff(x))))

def perm_entropy(x, m=3, tau=1):
    x = np.asarray(x); n = len(x)-(m-1)*tau
    if n<=0: return 0.0
    patterns={}
    for i in range(n):
        w = x[i:i+m*tau:tau]
        key = tuple(np.argsort(w))
        patterns[key] = patterns.get(key,0)+1
    cnt = np.array(list(patterns.values()), dtype=float)
    p = cnt / cnt.sum()
    return float(-np.sum(p*np.log(p+1e-12)))

def coherence_drop(true_seg, pred_seg, ch_names, fs, pairs=COH_PAIRS):
    name2idx = {nm:i for i,nm in enumerate(ch_names)}
    drops = []
    for a,b in pairs:
        if a in name2idx and b in name2idx:
            ta, tb = true_seg[name2idx[a]], true_seg[name2idx[b]]
            pa, pb = pred_seg[name2idx[a]], pred_seg[name2idx[b]]
            f, c_true = coherence(ta, tb, fs=fs, nperseg=min(128, len(ta)))
            _, c_pred = coherence(pa, pb, fs=fs, nperseg=min(128, len(pa)))
            m = (f>=8)&(f<=13)  # alpha
            ctru = float(np.mean(ensure_finite(c_true[m]))) if np.any(m) else 0.0
            cprd = float(np.mean(ensure_finite(c_pred[m]))) if np.any(m) else 0.0
            drops.append(max(0.0, ctru - cprd))
    return float(np.mean(drops) if drops else 0.0)

def pbi_for_window(y_true, y_pred, res, fs, ch_names, x_last=None):
    """
    y_true, y_pred: (C, H)
    res = y_true - y_pred
    x_last: (C, 1) — последний отсчёт истории. Если задан, считаем baseline 'naive hold'.
    """
    C, H = y_true.shape

    total_rmse = rmse(y_true, y_pred)
    total_mae  = mae(y_true, y_pred)
    rms_true   = np.sqrt(np.mean(y_true**2))
    denom      = max(0.1, rms_true)
    nrmse      = float(total_rmse / denom)

    lrmse = lag_rmse(y_true, y_pred, lags=[1,5,25,50])
    spec  = spectral_nrmse(y_true, y_pred, fs)

    res_flat = res.reshape(C*H)
    r1 = autocorr_lag1(res_flat)
    ll = line_length(res_flat)
    pe = perm_entropy(res_flat, m=3, tau=1)

    coh_drop = coherence_drop(y_true, y_pred, ch_names, fs)

    # Наивный baseline
    delta_nrmse = 0.0
    improve_rate = 0.0
    if x_last is not None:
        naive = np.repeat(x_last, H, axis=1)
        rmse_naive  = rmse(y_true, naive)
        nrmse_naive = float(rmse_naive / denom)
        delta_nrmse = float(nrmse_naive - nrmse)     # >0 → модель лучше наивной
        e_mod   = np.abs(y_true - y_pred)
        e_naive = np.abs(y_true - naive)
        improve_rate = float(np.mean(e_mod < e_naive))

    # Клип значений
    nrmse = float(min(nrmse, 8.0))
    total_mae = float(min(total_mae, 8.0))
    for band in list(BANDS.keys()):
        spec[band] = float(min(spec[band], 3.0))

    feats = {
        "nRMSE": nrmse, "MAE": total_mae,
        "RMSE_d1": lrmse.get(1,0.0), "RMSE_d5": lrmse.get(5,0.0),
        "RMSE_d25": lrmse.get(25,0.0), "RMSE_d50": lrmse.get(50,0.0),
        "spec_nrmse_delta": spec["delta"], "spec_nrmse_theta": spec["theta"],
        "spec_nrmse_alpha": spec["alpha"], "spec_nrmse_beta":  spec["beta"],
        "spec_nrmse_lowg":  spec["lowg"],
        "res_ac1": r1, "res_linelen": ll, "res_perm_entropy": pe,
        "coh_drop_alpha": coh_drop,
        "delta_nRMSE": delta_nrmse, "improve_rate": improve_rate
    }
    return feats

def aggregate_subject(pbi_list):
    """
    Возвращаем медианы + IQR по ключам.
    """
    keys = sorted(pbi_list[0].keys())
    agg = {}
    for k in keys:
        vals = np.array([d[k] for d in pbi_list], dtype=float)
        agg[f"{k}_med"] = float(np.nanmedian(vals))
        if k in ("nRMSE","delta_nRMSE","improve_rate",
                 "spec_nrmse_alpha","spec_nrmse_beta","spec_nrmse_lowg"):
            agg[f"{k}_iqr"] = float(IQR(vals))
    return agg
