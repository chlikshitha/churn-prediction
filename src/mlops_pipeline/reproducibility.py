"""Repeat deterministic local fits and record metric reproducibility."""
import json
from pathlib import Path
import joblib, numpy as np, pandas as pd, yaml
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
ROOT=Path(__file__).resolve().parents[2]
def main():
    cfg=yaml.safe_load((ROOT/"config.yaml").read_text(encoding="utf-8")); d=ROOT/cfg["data"]["processed_dir"]
    Xtr=pd.read_csv(d/"X_train_raw.csv"); Xte=pd.read_csv(d/"X_test_raw.csv"); ytr=np.load(d/"y_train.npy"); yte=np.load(d/"y_test.npy")
    model=joblib.load(ROOT/cfg["paths"]["full_pipeline"]); preds=model.predict(Xte); probs=model.predict_proba(Xte)[:,1]
    metrics={"accuracy":float(accuracy_score(yte,preds)),"f1_score":float(f1_score(yte,preds,zero_division=0)),"roc_auc":float(roc_auc_score(yte,probs))}
    from sklearn.base import clone
    repeated=clone(model).fit(Xtr,ytr); pred2=repeated.predict(Xte); prob2=repeated.predict_proba(Xte)[:,1]
    repeated_metrics={"accuracy":float(accuracy_score(yte,pred2)),"f1_score":float(f1_score(yte,pred2,zero_division=0)),"roc_auc":float(roc_auc_score(yte,prob2))}
    passed=all(abs(metrics[k]-repeated_metrics[k])<1e-12 for k in metrics)
    report={"runs":[metrics,repeated_metrics],"reproducible":passed,"status":"PASSED" if passed else "FAILED"}
    out=ROOT/"artifacts"; out.mkdir(exist_ok=True)
    for name in ["reproducibility_report.json","pipeline_reproducibility_report.json"]:
        (out/name).write_text(json.dumps(report,indent=2),encoding="utf-8")
    if not passed: raise SystemExit("Repeated metrics do not match")
if __name__=="__main__": main()
