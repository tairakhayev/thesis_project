# detect_anomaly_windows_loso.py
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, balanced_accuracy_score, f1_score,
    confusion_matrix, roc_curve
)

DEFAULT_CSV = os.path.join(os.getcwd(), "pbi_windows.csv")

def aggregate_scores(win_scores: np.ndarray, method: str = "q90", topk_frac: float = 0.1) -> float:
    """Агрегация оконных аномалий в один пациентский скор (больше = аномальнее)."""
    if win_scores.size == 0:
        return float("nan")
    if method == "q90":
        return float(np.quantile(win_scores, 0.90))
    elif method == "topk":
        k = max(5, int(np.ceil(topk_frac * len(win_scores))))
        return float(np.mean(np.sort(win_scores)[-k:]))
    elif method == "median":
        return float(np.median(win_scores))
    elif method == "mean":
        return float(np.mean(win_scores))
    else:
        raise ValueError(f"Unknown aggregation method: {method}")

def summarize_patient_window_scores(win_scores: np.ndarray, topk_frac: float = 0.1) -> dict:
    """Набор patient-level фичей из оконных скорингов."""
    if win_scores.size == 0:
        return {"score_q90": np.nan, "score_topk": np.nan, "score_mean": np.nan,
                "score_median": np.nan, "score_var": np.nan}
    k = max(5, int(np.ceil(topk_frac * len(win_scores))))
    return {
        "score_q90":    float(np.quantile(win_scores, 0.90)),
        "score_topk":   float(np.mean(np.sort(win_scores)[-k:])),
        "score_mean":   float(np.mean(win_scores)),
        "score_median": float(np.median(win_scores)),
        "score_var":    float(np.var(win_scores)),
    }

def fit_iforest(Xtr_z: np.ndarray, n_estimators: int, max_samples: int, contamination, random_state: int) -> IsolationForest:
    iforest = IsolationForest(
        n_estimators=n_estimators,
        max_samples=max_samples,
        contamination=contamination,
        bootstrap=False,
        random_state=random_state,
        n_jobs=-1,
        warm_start=False
    )
    iforest.fit(Xtr_z)
    return iforest

def save_plots(y_true, prob, pred, cm, out_dir, tag="iforest"):
    os.makedirs(out_dir, exist_ok=True)

    # ROC
    fpr, tpr, _ = roc_curve(y_true, prob)
    auc = roc_auc_score(y_true, prob)
    plt.figure(figsize=(5,4))
    plt.plot(fpr, tpr, label=f"AUC={auc:.3f}")
    plt.plot([0,1],[0,1], '--', color='gray')
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{tag}_roc.png"))
    plt.close()

    # Confusion matrix
    plt.figure(figsize=(4.5,4))
    im = plt.imshow(cm, cmap="Blues")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    for (i,j), v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha='center', va='center')
    plt.xticks([0,1], ["Pred 0","Pred 1"])
    plt.yticks([0,1], ["True 0","True 1"])
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{tag}_cm.png"))
    plt.close()

    # Distributions
    prob = np.asarray(prob)
    y_true = np.asarray(y_true)
    plt.figure(figsize=(6,4))
    plt.hist(prob[y_true==0], bins=12, alpha=0.6, label="Healthy", density=True)
    plt.hist(prob[y_true==1], bins=12, alpha=0.6, label="Sick", density=True)
    plt.xlabel("patient score (prob)"); plt.ylabel("density"); plt.title("Patient scores")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{tag}_score_dist.png"))
    plt.close()

def main(args):
    df = pd.read_csv(args.csv)
    meta_cols = ["subject", "true", "win_idx"]
    feats = [c for c in df.columns if c not in meta_cols]

    subjects = df["subject"].unique().tolist()
    oof_patient_score, y_patient, subj_list, n_windows, extra_feats = [], [], [], [], []

    print(f"Loaded: {args.csv} | subjects={len(subjects)} | feat_dim={len(feats)}")

    for s in subjects:
        te_mask = (df["subject"] == s)
        tr_mask = ~te_mask

        # semi-supervised: учим IF только на healthy train-окнах (если их >=200)
        healthy_tr_mask = (df.loc[tr_mask, "true"].values == 0)
        Xtr = df.loc[tr_mask, feats].values
        Xte = df.loc[te_mask, feats].values
        yte = int(df.loc[te_mask, "true"].iloc[0])

        if healthy_tr_mask.sum() >= 200:
            Xtr = Xtr[healthy_tr_mask]

        scaler = StandardScaler().fit(Xtr)
        Xtr_z = scaler.transform(Xtr)
        Xte_z = scaler.transform(Xte)

        contam = args.contamination if args.contamination == "auto" else float(args.contamination)
        iso = fit_iforest(
            Xtr_z,
            n_estimators=args.n_estimators,
            max_samples=args.max_samples,
            contamination=contam,
            random_state=args.seed,
        )

        # окно-скор: -decision_function -> БОЛЬШЕ = более аномально
        win_score = -iso.decision_function(Xte_z)

        # агрегаты (для последующей классификации)
        feats_dict = summarize_patient_window_scores(win_score, topk_frac=args.topk_frac)
        extra_feats.append(feats_dict)

        # основной patient-score по выбранной агрегации
        patient_score = aggregate_scores(win_score, method=args.agg, topk_frac=args.topk_frac)

        oof_patient_score.append(patient_score)
        y_patient.append(yte)
        subj_list.append(s)
        n_windows.append(len(win_score))

        print(f"[{s}] true={yte} | iforest_{args.agg}={patient_score:.3f} | n_win={len(win_score)}")

    oof_patient_score = np.asarray(oof_patient_score, dtype=float)
    y_patient = np.asarray(y_patient, dtype=int)

    # вероятности
    ps_minmax = (oof_patient_score - np.nanmin(oof_patient_score)) / (np.nanmax(oof_patient_score) - np.nanmin(oof_patient_score) + 1e-9)
    ranks = pd.Series(oof_patient_score).rank(method="average").to_numpy()
    ps_rank = (ranks - 1) / (len(ranks) - 1 + 1e-9)

    # используем одну переменную prob в дальнейших расчётах
    prob = ps_minmax

    # выбор порога: либо ограничение на FPR, либо классический Youden J
    fpr, tpr, thr = roc_curve(y_patient, prob)
    if args.target_fpr is not None:
        target = float(args.target_fpr)
        mask = fpr <= target + 1e-12
        if np.any(mask):
            idx = np.argmax(tpr[mask])
            thr_star = float(thr[mask][idx])
        else:
            j = tpr - fpr
            thr_star = float(thr[np.argmax(j)])
    else:
        j = tpr - fpr
        thr_star = float(thr[np.argmax(j)])

    pred = (prob >= thr_star).astype(int)

    auc = roc_auc_score(y_patient, prob)
    bal = balanced_accuracy_score(y_patient, pred)
    f1  = f1_score(y_patient, pred)
    cm  = confusion_matrix(y_patient, pred, labels=[0, 1])

    print("\n=== Patient-level (IsolationForest) ===")
    print(f"AUC={auc:.3f} | BalancedAcc={bal:.3f} | F1={f1:.3f} | thr*={thr_star:.3f}")
    print(cm)

    # расширенный per-subject результат
    out = pd.DataFrame({
        "subject": subj_list,
        "true": y_patient,
        "score_raw": oof_patient_score,
        "prob_sick": prob,             # min-max
        "prob_sick_rank": ps_rank,     # ранговая
        "n_windows": n_windows,
        "agg": args.agg,
        "n_estimators": args.n_estimators,
        "max_samples": args.max_samples,
        "contamination": args.contamination
    })

    out = pd.concat([out, pd.DataFrame(extra_feats)], axis=1).sort_values("subject")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"\nSaved -> {args.out}")

    if args.save_plots:
        save_plots(y_patient, prob, pred, cm, args.plot_dir, tag="iforest_loso")
        print(f"Plots saved to: {args.plot_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LOSO anomaly detection with IsolationForest over window features.")
    parser.add_argument("--csv", type=str, default=DEFAULT_CSV, help="Path to pbi_windows.csv")
    parser.add_argument("--out", type=str, default="detect_anomaly_loso_results.csv", help="Where to save per-subject scores")
    parser.add_argument("--agg", type=str, default="q90", choices=["q90", "topk", "median", "mean"], help="Patient-level aggregation")
    parser.add_argument("--topk_frac", type=float, default=0.10, help="Top-k fraction for 'topk' aggregation")
    parser.add_argument("--n_estimators", type=int, default=600, help="IsolationForest trees")
    parser.add_argument("--max_samples", type=int, default=512, help="IsolationForest max_samples")
    parser.add_argument("--contamination", type=str, default="auto", help="'auto' or float in (0, 0.5]")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--target_fpr", type=float, default=None, help="Если задано, выбираем порог с FPR <= target_fpr (иначе Youden J)")
    parser.add_argument("--save_plots", action="store_true", help="Save ROC/CM/score hist to plots dir")
    parser.add_argument("--plot_dir", type=str, default=os.path.join("final","plots"), help="Where to save plots")

    args = parser.parse_args()
    main(args)
