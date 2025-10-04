# classify_patient_supervised.py
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve,
    confusion_matrix, balanced_accuracy_score, f1_score
)
from sklearn.linear_model import LogisticRegressionCV


def iqr(x):
    x = np.asarray(x, dtype=float)
    q75, q25 = np.nanpercentile(x, 75), np.nanpercentile(x, 25)
    return float(q75 - q25)

def aggregate_pbi_patient_level(pbi_df):
    meta = ["subject", "true", "win_idx"]
    feats = [c for c in pbi_df.columns if c not in meta]
    rows = []
    for subj, g in pbi_df.groupby("subject"):
        y = int(g["true"].iloc[0])
        rec = {"subject": subj, "true": y}
        for f in feats:
            vals = g[f].values
            rec[f"{f}_med"] = float(np.nanmedian(vals))
            rec[f"{f}_iqr"] = iqr(vals)
        rows.append(rec)
    return pd.DataFrame(rows)

def plot_and_save_roc_pr_cm(y_true, y_prob, y_pred, out_dir, tag="supervised"):
    os.makedirs(out_dir, exist_ok=True)
    # ROC
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    plt.figure(figsize=(5,4))
    plt.plot(fpr, tpr, label=f"AUC={auc:.3f}")
    plt.plot([0,1],[0,1],'--')
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{tag}_roc.png")); plt.close()
    # PR
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    plt.figure(figsize=(5,4))
    plt.plot(rec, prec)
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("PR curve")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{tag}_pr.png")); plt.close()
    # CM
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    plt.figure(figsize=(4.5,4))
    im = plt.imshow(cm, cmap="Blues")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    for (i,j), v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha='center', va='center')
    plt.xticks([0,1], ["Pred 0","Pred 1"])
    plt.yticks([0,1], ["True 0","True 1"])
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{tag}_cm.png")); plt.close()
    return auc, cm

def main(args):
    # 1) PBI окна -> пациент
    pbi = pd.read_csv(args.pbi_csv)
    pbi_pat = aggregate_pbi_patient_level(pbi)

    # 2) IsolationForest patient-level (1 или 2 файлов)
    if_df = pd.read_csv(args.iforest_csv)
    if args.iforest_csv2:
        if2 = pd.read_csv(args.iforest_csv2)
        clash = [c for c in if2.columns if c in if_df.columns and c not in ("subject","true")]
        if2 = if2.rename(columns={c: f"{c}_b" for c in clash})
        if_df = pd.merge(if_df, if2, on=["subject","true"], how="inner")

    # 3) merge
    df = pd.merge(pbi_pat, if_df, on=["subject","true"], how="inner")

    # признаки: все patient-агрегаты PBI + IF-поля
    # признаки: все patient-агрегаты PBI + IF-поля
    drop_cols = {
        "subject","true","win_idx",
        "agg","agg_b",                # текстовые
        "contamination","contamination_b"  # может быть строкой ("auto")
    }

    # кандидаты
    X_cols = [c for c in df.columns if c not in drop_cols and not c.endswith("_raw")]

    # обязательно включим базовые IF-фичи, если они есть в таблице
    must_if = ["score_q90","score_topk","score_mean","score_median","score_var",
               "prob_sick","prob_sick_rank","n_windows"]
    X_cols = sorted(set(X_cols).union(must_if).intersection(df.columns))

    # оставляем только числовые столбцы (чтобы исключить любые неожиданно-строковые)
    num_cols = df[X_cols].select_dtypes(include=[np.number]).columns.tolist()
    dropped = sorted(set(X_cols) - set(num_cols))
    if dropped:
        print("Dropping non-numeric cols:", dropped)

    def drop_high_corr(df_num, thr=0.90):
        corr = df_num.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        to_drop = [col for col in upper.columns if any(upper[col] > thr)]
        return df_num.drop(columns=to_drop), to_drop

    X_df = df[num_cols].copy()
    X_df, hi_drop = drop_high_corr(X_df, thr=0.95)
    if hi_drop:
        print(f"Drop highly correlated ({len(hi_drop)}):", hi_drop)

    used_cols = X_df.columns.tolist()
    X = X_df.to_numpy(dtype=float)
    y = df["true"].to_numpy(dtype=int)
    N = len(df)

    # 4) модели как Pipelines (без утечек)
    models = {
        "LR":   make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, solver="lbfgs", random_state=42)),
        "LRcv": make_pipeline(StandardScaler(), LogisticRegressionCV(
                    Cs=np.logspace(-3, 2, 20), cv=5, scoring="roc_auc",
                    max_iter=5000, solver="lbfgs", n_jobs=-1, refit=True)),
        "LR_L1": make_pipeline(StandardScaler(), LogisticRegressionCV(
                    Cs=np.logspace(-3, 2, 20), cv=5, scoring="roc_auc",
                    penalty="l1", solver="saga", max_iter=5000, n_jobs=-1, refit=True)),
        "RF":   make_pipeline(StandardScaler(), RandomForestClassifier(
                    n_estimators=400, max_depth=None, random_state=42)),
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}

    for name, pipeline in models.items():
        # OOF-контейнеры по индексам, чтобы не путать порядок
        oof_prob = np.zeros(N, dtype=float)
        oof_pred = np.zeros(N, dtype=int)

        for tr, te in skf.split(X, y):
            pipeline.fit(X[tr], y[tr])

            if hasattr(pipeline[-1], "predict_proba"):
                prob = pipeline.predict_proba(X[te])[:,1]
            else:
                s = pipeline.decision_function(X[te])
                prob = (s - s.min())/(s.max() - s.min() + 1e-9)

            # порог по Youden J на текущем валидационном фолде
            fpr, tpr, thr = roc_curve(y[te], prob)
            j = tpr - fpr
            thr_star = float(thr[np.argmax(j)])
            pred = (prob >= thr_star).astype(int)

            oof_prob[te] = prob
            oof_pred[te] = pred

        auc = roc_auc_score(y, oof_prob)
        bal = balanced_accuracy_score(y, oof_pred)
        f1  = f1_score(y, oof_pred)
        results[name] = {"auc": auc, "bal": bal, "f1": f1,
                         "y_true": y, "y_prob": oof_prob, "y_pred": oof_pred}

        print(f"\n=== {name} ===")
        print(f"AUC={auc:.3f} | BalancedAcc={bal:.3f} | F1={f1:.3f}")

    # 5) лучшая по AUC
    best_name = max(results, key=lambda k: results[k]["auc"])
    best = results[best_name]
    print(f"\n>>> BEST by AUC: {best_name} | AUC={best['auc']:.3f} | BalAcc={best['bal']:.3f} | F1={best['f1']:.3f}")

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    out_df = pd.DataFrame({
        "subject": df["subject"],
        "true": df["true"],
        "prob_sick": best["y_prob"],
        "pred": best["y_pred"]
    })
    out_df.to_csv(args.out_csv, index=False)
    print(f"Saved -> {args.out_csv}")

    # графики
    auc_val, cm = plot_and_save_roc_pr_cm(best["y_true"], best["y_prob"], best["y_pred"], args.plot_dir, tag=f"supervised_{best_name}")
    print(f"Plots saved to: {args.plot_dir}")
    print("Confusion matrix:\n", cm)
    
        # --- Интерпретация: важности признаков для лучшей LR-модели ---
    if best_name in {"LR", "LRcv", "LR_L1"}:
        # переобучаем ту же модель на всех данных для интерпретации
        pipe = models[best_name]
        pipe.fit(X, y)

        # имена использованных фич (после фильтра корреляций)
        feat_names = used_cols  # сформированы выше при сборке X_df

        # коэффициенты LR (на стандартизированных фичах)
        coef = pipe[-1].coef_.ravel()
        imp = pd.Series(coef, index=feat_names).sort_values(key=np.abs, ascending=False)

        os.makedirs("final", exist_ok=True)
        imp.to_csv(os.path.join("final", f"{best_name}_feature_importance.csv"))

        # топ-15 барчарт
        topk = 15
        top_imp = imp.iloc[:topk][::-1]  # для красивого вертикального порядка
        plt.figure(figsize=(8, 6))
        plt.barh(top_imp.index, top_imp.values)
        plt.title(f"Top-{topk} LR coefficients ({best_name})")
        plt.tight_layout()
        plt.savefig(os.path.join(args.plot_dir, f"{best_name}_feature_importance.png"))
        plt.close()
        print(f"Saved feature importances to final/{best_name}_feature_importance.csv and plot to {args.plot_dir}/{best_name}_feature_importance.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Supervised classification on patient-level PBI + IF features.")
    parser.add_argument("--pbi_csv", type=str, default="pbi_windows.csv")
    parser.add_argument("--iforest_csv", type=str, required=True, help="Output of detect_anomaly_windows_loso.py")
    parser.add_argument("--iforest_csv2", type=str, default=None, help="Optional second IF CSV for extra patient-level features")
    parser.add_argument("--out_csv", type=str, default=os.path.join("final","classify_supervised_results.csv"))
    parser.add_argument("--plot_dir", type=str, default=os.path.join("final","plots"))
    args = parser.parse_args()
    main(args)
