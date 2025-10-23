# src/train_sklearn.py
import argparse, os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from src.data import load_dataset, split_stratified, LABEL_ORDER, label_to_id
from src.metrics import macro_f1, full_report, confmat
from src.utils import save_json, plot_confusion

def run(model_name: str, C: float = 1.0, seed: int = 42):
    df = load_dataset()
    tr, va, te = split_stratified(df, seed=seed)
    y_train = tr["label"].map(label_to_id).values
    y_val   = va["label"].map(label_to_id).values
    y_test  = te["label"].map(label_to_id).values

    vec = TfidfVectorizer(ngram_range=(1,2), min_df=2, max_features=100_000)
    if model_name == "logreg":
        cls = LogisticRegression(C=C, max_iter=500, class_weight="balanced", n_jobs=1)
    elif model_name == "linearsvm":
        cls = LinearSVC(C=C, class_weight="balanced")
    else:
        raise ValueError("model_name must be logreg or linearsvm")

    pipe = Pipeline([("tfidf", vec), ("clf", cls)])
    pipe.fit(tr["text"], y_train)

    results = {}
    for name, X, y in [("val", va["text"], y_val), ("test", te["text"], y_test)]:
        pred = pipe.predict(X)
        f1 = macro_f1(y, pred)
        rep = full_report(y, pred, LABEL_ORDER)
        cm = confmat(y, pred, LABEL_ORDER)
        results[name] = {"macro_f1": float(f1), "report": rep, "cm": cm.tolist()}
        plot_confusion(cm, LABEL_ORDER, f"artifacts/baselines/{model_name}_{name}_cm.png")

    os.makedirs("artifacts/baselines", exist_ok=True)
    save_json(results, f"artifacts/baselines/{model_name}_results.json")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["logreg","linearsvm"], required=True)
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    run(args.model, args.C, args.seed)
