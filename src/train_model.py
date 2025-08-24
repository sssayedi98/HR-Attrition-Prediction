"""HR Attrition Prediction — real-data trainer
Attempts to download IBM HR Attrition dataset from common public mirrors.
If download fails, falls back to sample_hr_attrition.csv.
"""
import pandas as pd, numpy as np, matplotlib.pyplot as plt, json
from pathlib import Path
from urllib.request import urlopen, Request
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix, RocCurveDisplay, PrecisionRecallDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

DATA = Path("data"); DATA.mkdir(exist_ok=True)
RESULTS = Path("results"); RESULTS.mkdir(exist_ok=True)

URLS = [
    "https://raw.githubusercontent.com/IBM/employee-attrition-aif360/master/data/WA_Fn-UseC_-HR-Employee-Attrition.csv",
    "https://raw.githubusercontent.com/rs-dev29/ibm-hr-analytics-attrition/master/WA_Fn-UseC_-HR-Employee-Attrition.csv"
]

def download_csv(urls, out_path):
    last_err=None
    for u in urls:
        try:
            req = Request(u, headers={"User-Agent":"Mozilla/5.0"})
            with urlopen(req, timeout=60) as resp:
                data = resp.read()
            open(out_path, "wb").write(data)
            print("Downloaded:", u)
            return True
        except Exception as e:
            print("Failed:", u, e); last_err=e
    print("All downloads failed.", last_err); return False

raw = DATA / "hr_attrition_raw.csv"
if not raw.exists():
    ok = download_csv(URLS, raw)
    if not ok:
        print("Falling back to sample_hr_attrition.csv")
        raw = DATA / "sample_hr_attrition.csv"

df = pd.read_csv(raw)
target_col = "Attrition" if "Attrition" in df.columns else "attrition"
y = (df[target_col].astype(str).str.strip().str.lower()=="yes").astype(int)
X = df.drop(columns=[target_col], errors="ignore")

num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
cat_cols = [c for c in X.columns if c not in num_cols]

pre = ColumnTransformer([("num", StandardScaler(), num_cols), ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)])
log_reg = Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))])
rf = Pipeline([("pre", pre), ("clf", RandomForestClassifier(n_estimators=400, random_state=42, class_weight="balanced"))])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

log_reg.fit(X_train, y_train); rf.fit(X_train, y_train)
proba_rf = rf.predict_proba(X_test)[:,1]; pred_rf = (proba_rf>=0.5).astype(int)
proba_lr = log_reg.predict_proba(X_test)[:,1]; pred_lr = (proba_lr>=0.5).astype(int)

def metrics_block(y_true, y_pred, y_proba):
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "support_positive": int((y_true==1).sum())
    }

metrics = {"baseline_attrition_rate": float(y.mean()), "random_forest": metrics_block(y_test, pred_rf, proba_rf), "logistic_regression": metrics_block(y_test, pred_lr, proba_lr)}
json.dump(metrics, open(RESULTS / "metrics_real.json","w"), indent=2)

# Plots
cm = confusion_matrix(y_test, pred_rf)
plt.figure(); plt.imshow(cm, cmap="Blues"); plt.title("Confusion Matrix — RF (Real Data)"); plt.xlabel("Predicted"); plt.ylabel("Actual")
for (i,j), v in np.ndenumerate(cm): plt.text(j,i,str(v),ha="center",va="center")
plt.tight_layout(); plt.savefig(RESULTS / "confusion_matrix_rf_real.png", dpi=300); plt.close()

plt.figure(); RocCurveDisplay.from_predictions(y_test, proba_rf); plt.title("ROC Curve — RF (Real Data)"); plt.tight_layout(); plt.savefig(RESULTS / "roc_curve_rf_real.png", dpi=300); plt.close()
plt.figure(); PrecisionRecallDisplay.from_predictions(y_test, proba_rf); plt.title("PR Curve — RF (Real Data)"); plt.tight_layout(); plt.savefig(RESULTS / "pr_curve_rf_real.png", dpi=300); plt.close()

scored = pd.DataFrame({"attrition_proba_rf": proba_rf}).sort_values("attrition_proba_rf", ascending=False)
scored.to_csv(RESULTS / "attrition_risk_scored_real.csv", index=False)

print("Training complete. Artifacts in results/.")