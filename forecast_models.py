# forecast_models.py
import math, random, numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ----------------------- utils -----------------------
def set_seed(seed=42):
    np.random.seed(seed); random.seed(seed); torch.manual_seed(seed)

class Seq2SeqWindows(Dataset):
    def __init__(self, X_windows, Y_windows):
        self.X = X_windows.astype(np.float32)   # (N, C, L)
        self.Y = Y_windows.astype(np.float32)   # (N, C, H)
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, i): return self.X[i], self.Y[i]

# ------------------- classical AR --------------------
class ARBaseline:
    """Простой многошаговый AR-регрессор по последним p отсчётам (per-channel)."""
    def __init__(self, p=8):
        self.p = int(p)
        self.coef_ = None  # (C, p, H)
        self.bias_ = None  # (C, H)

    def fit(self, X, Y):
        # X:(N,C,L), Y:(N,C,H) — используем последние p лагов
        N,C,L = X.shape; H = Y.shape[2]; p = min(self.p, L)
        Phi_all = np.stack([X[:,:,L-k-1] for k in range(p)], axis=2)  # (N,C,p)
        Tgt_all = Y.transpose(0,1,2)                                  # (N,C,H)
        coef = np.zeros((C,p,H), np.float32); bias = np.zeros((C,H), np.float32)
        for c in range(C):
            phi = Phi_all[:,c,:]                                      # (N,p)
            phi1 = np.concatenate([phi, np.ones((phi.shape[0],1),np.float32)], axis=1)
            tgt = Tgt_all[:,c,:]                                      # (N,H)
            W, _, _, _ = np.linalg.lstsq(phi1, tgt, rcond=None)       # (p+1,H)
            coef[c] = W[:-1,:]; bias[c] = W[-1,:]
        self.coef_, self.bias_ = coef, bias
        return self

    def predict(self, X):
        N,C,L = X.shape; p = min(self.p, L); H = self.bias_.shape[1]
        Phi = np.stack([X[:,:,L-k-1] for k in range(p)], axis=2)      # (N,C,p)
        Yhat = np.einsum('ncp,cph->nch', Phi, self.coef_) + self.bias_[None,:,:]
        return Yhat.astype(np.float32)

# --------------- TCN building blocks -----------------
class TemporalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=5, dilation=1, dropout=0.1):
        super().__init__()
        pad = (kernel_size-1)*dilation
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad, dilation=dilation)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad, dilation=dilation)
        self.relu2 = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.down = nn.Conv1d(in_ch, out_ch, 1) if in_ch!=out_ch else None

    def forward(self, x):
        out = self.conv1(x); out = self.relu1(out)
        out = self.conv2(out); out = self.drop(self.relu2(out))
        res = x if self.down is None else self.down(x)
        # causal cut: убираем "look-ahead" паддинг
        cut = out.shape[-1] - x.shape[-1]
        if cut>0: out = out[:, :, :-cut]
        return out + res

class TCNForecaster(nn.Module):
    """
    FIXED: корректная multi-horizon голова.
    Берём последний скрытый вектор и через Linear(hid -> C*H) предсказываем все H шагов.
    """
    def __init__(self, C, H, n_blocks=4, hid=64, k=5, dropout=0.1, predict_sigma=False):
        super().__init__()
        layers = []
        in_ch = C
        for b in range(n_blocks):
            layers.append(TemporalBlock(in_ch, hid, kernel_size=k, dilation=2**b, dropout=dropout))
            in_ch = hid
        self.tcn = nn.Sequential(*layers)
        self.proj_mu = nn.Linear(hid, C*H)
        self.predict_sigma = predict_sigma
        if predict_sigma:
            self.proj_lv = nn.Linear(hid, C*H)
        self.C, self.H = C, H

    def forward(self, x):            # x: (B,C,L)
        h = self.tcn(x)              # (B,hid,L)
        h_last = h[:, :, -1]         # (B,hid)
        mu = self.proj_mu(h_last).view(-1, self.C, self.H)
        if self.predict_sigma:
            lv = self.proj_lv(h_last).view(-1, self.C, self.H)
            return mu, lv
        return mu, None

def gaussian_nll(y, mu, logvar, eps=1e-6):
    var = (logvar.exp() + eps)
    return 0.5*((y-mu)**2/var + logvar).mean()

def train_tcn(X_tr, Y_tr, X_va, Y_va, epochs=30, bs=128, lr=3e-4, device='cpu', predict_sigma=False):
    """Тренер для TCNForecaster (без AR2-обёртки)."""
    tr_ds = Seq2SeqWindows(X_tr, Y_tr); va_ds = Seq2SeqWindows(X_va, Y_va)
    tr = DataLoader(tr_ds, batch_size=bs, shuffle=True)
    va = DataLoader(va_ds, batch_size=bs, shuffle=False)
    C, H = X_tr.shape[1], Y_tr.shape[2]
    model = TCNForecaster(C, H, n_blocks=4, hid=64, k=5, dropout=0.1, predict_sigma=predict_sigma).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best = math.inf; best_state=None; stale=0
    for ep in range(1, epochs+1):
        model.train(); loss_tr=0.0; n=0
        for xb, yb in tr:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            mu, lv = model(xb)
            loss = ((yb-mu)**2).mean() if lv is None else gaussian_nll(yb, mu, lv)
            loss.backward(); opt.step()
            loss_tr += loss.item()*xb.size(0); n += xb.size(0)
        model.eval(); loss_va=0.0; n2=0
        with torch.no_grad():
            for xb, yb in va:
                xb, yb = xb.to(device), yb.to(device)
                mu, lv = model(xb)
                loss = ((yb-mu)**2).mean() if lv is None else gaussian_nll(yb, mu, lv)
                loss_va += loss.item()*xb.size(0); n2 += xb.size(0)
        val = loss_va/max(1,n2)
        if val < best:
            best = val; best_state = {k:v.cpu().clone() for k,v in model.state_dict().items()}; stale=0
        else:
            stale += 1
        if stale>=6: break
    if best_state is not None: model.load_state_dict(best_state)
    return model

# ------------------ Strong linear baseline ------------------
class DLinearForecaster(nn.Module):
    """
    Лёгкий и сильный baseline: два линейных слоя «история -> будущее».
    Работает по-канально (без перемешивания каналов).
    """
    def __init__(self, C, L, H):
        super().__init__()
        self.C, self.L, self.H = C, L, H
        self.trend  = nn.Linear(L, H, bias=True)
        self.season = nn.Linear(L, H, bias=True)

    def forward(self, x):        # x: (B,C,L)
        # применяем линейки ко всем каналам независимо
        t = self.trend(x)        # (B,C,H)
        s = self.season(x)       # (B,C,H)
        return t + s

# ------------------- AR(2) utilities -------------------
def _fit_ar2_from_tail(x, M=150):
    M = min(M, len(x)-2)
    if M < 10: return 1.0, 0.0
    y = x[-M:]; Y = y[2:]; Phi = np.stack([y[1:-1], y[:-2]], axis=1)
    lam = 1e-3
    A = Phi.T @ Phi + lam*np.eye(2, dtype=np.float32)
    b = Phi.T @ Y
    a = np.linalg.solve(A, b)
    return float(a[0]), float(a[1])

def _ar2_forecast(history, H, M=150):
    C, _ = history.shape
    out = np.zeros((C,H), dtype=history.dtype)
    for c in range(C):
        x = history[c]; a1, a2 = _fit_ar2_from_tail(x, M)
        y1, y2 = x[-1], x[-2]
        for t in range(H):
            y = a1*y1 + a2*y2
            out[c, t] = y
            y2, y1 = y1, y
    return out

# -------------- Residual-to-AR(2) wrapper --------------
class ResidualToAR2(nn.Module):
    """
    Оборачивает любую модель f: (B,C,L)->(B,C,H) так, что итог = AR2(history) + f(history).
    Дополнительно делает мягкий клип относительно последнего значения.
    """
    def __init__(self, base_model, M=150, clip_k=4.0):
        super().__init__()
        self.base = base_model
        self.M = M
        self.clip_k = clip_k
        # ожидаем, что у base_model есть атрибуты C, H (как у TCN/DLinear ниже)
        self.C = getattr(base_model, 'C', None)
        self.H = getattr(base_model, 'H', None)

    def forward(self, x):  # x: (B,C,L), torch
        B, C, L = x.shape
        H = self.H
        assert H is not None, "base_model must expose .H"

        # baseline AR2 (numpy → torch)
        x_np = x.detach().cpu().numpy()
        Yb = np.stack([_ar2_forecast(x_np[i], H, M=self.M) for i in range(B)], axis=0)
        Yb = torch.from_numpy(Yb).to(x.device).float()                # (B,C,H)

        d_hat = self.base(x)                                          # (B,C,H) or (mu, lv)
        if isinstance(d_hat, tuple): d_hat = d_hat[0]
        y_hat = Yb + d_hat                                            # absolute

        # мягкий клип
        last = x[:, :, -1:].repeat(1,1,H)
        hstd = x.std(dim=2, keepdim=True).clamp_min(1e-6).repeat(1,1,H)
        y_hat = torch.max(torch.min(y_hat, last + self.clip_k*hstd), last - self.clip_k*hstd)
        return y_hat

# ---------------- Linear ridge forecasters ----------------
class RidgeHankelForecaster:
    """
    Обучение: (X^T X + λI) W = X^T Y
    X: (N, C*L) — последние L отсчётов всех C каналов
    Y: (N, C*H) — будущие H отсчётов всех каналов
    """
    def __init__(self, lam=1e-3):
        self.lam = float(lam)
        self.W = None   # (C*L+1, C*H)

    def _design(self, Xw):  # Xw: (N, C, L)
        N, C, L = Xw.shape
        X = Xw.reshape(N, C*L)
        X = np.concatenate([X, np.ones((N,1),dtype=np.float32)], axis=1)  # + bias
        return X

    def fit(self, X_windows, Y_windows):  # (N,C,L), (N,C,H)
        X = self._design(X_windows)
        Y = Y_windows.reshape(Y_windows.shape[0], -1)  # (N, C*H)
        XtX = X.T @ X
        d = XtX.shape[0]
        A = XtX + self.lam * np.eye(d, dtype=np.float32)
        self.W = np.linalg.solve(A, X.T @ Y).astype(np.float32)
        return self

    def predict(self, X_windows):  # (N,C,L) -> (N,C,H)
        X = self._design(X_windows)
        Yhat = X @ self.W
        N = X_windows.shape[0]; C = X_windows.shape[1]
        H = Yhat.shape[1] // C
        return Yhat.reshape(N, C, H).astype(np.float32)

class ResidualRidgeHankelForecaster:
    """
    Прогнозирует ΔY = Y - last(X). Опционально нормирует по std истории.
    """
    def __init__(self, lam=1e-3, norm_by_hist_std=True):
        self.lam = float(lam)
        self.norm_by_hist_std = bool(norm_by_hist_std)
        self.W = None  # (C*L+1, C*H)

    def _prep(self, Xw, Yw):
        N, C, L = Xw.shape
        H = Yw.shape[2]
        last = Xw[:, :, -1][:, :, None]         # (N,C,1)
        dY = Yw - last                          # (N,C,H)

        if self.norm_by_hist_std:
            s = np.std(Xw, axis=2, keepdims=True) + 1e-6
            Xn = (Xw - last) / s
            dYn = dY / s
            scale = s
        else:
            Xn = Xw; dYn = dY; scale = np.ones_like(last)

        X = np.concatenate([Xn.reshape(N, C*L), np.ones((N,1), np.float32)], axis=1)
        Y = dYn.reshape(N, C*H)
        return X, Y, last, scale

    def fit(self, Xw, Yw):
        X, Y, _, _ = self._prep(Xw, Yw)
        XtX = X.T @ X
        A = XtX + self.lam * np.eye(XtX.shape[0], dtype=np.float32)
        self.W = np.linalg.solve(A, X.T @ Y).astype(np.float32)
        return self

    def predict(self, Xw, Yw_ref=None):
        N, C, L = Xw.shape
        if Yw_ref is None:
            assert self.W is not None, "Call fit() before predict() or pass Yw_ref"
            H = self.W.shape[1] // C
            Yw_ref = np.zeros((N, C, H), dtype=np.float32)
        X, _, last, scale = self._prep(Xw, Yw_ref)
        dYhat = (X @ self.W).reshape(N, C, -1)
        Yhat = last + scale * dYhat
        return Yhat.astype(np.float32)

# -------------------- factories --------------------
def make_tcn(C, L, H, residual_to_ar2=True, **kwargs):
    """
    Создаёт TCN (при необходимости — с обёрткой ResidualToAR2).
    Возвращает torch.nn.Module, у которого .H доступен.
    """
    base = TCNForecaster(C, H, **kwargs)
    base.C, base.H = C, H
    return ResidualToAR2(base) if residual_to_ar2 else base

def make_dlinear(C, L, H, residual_to_ar2=True):
    base = DLinearForecaster(C, L, H)
    base.C, base.H = C, H
    return ResidualToAR2(base) if residual_to_ar2 else base
