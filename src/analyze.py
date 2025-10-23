import json, glob, os, pandas as pd

def load_json(p):
    with open(p) as f: return json.load(f)

def main():
    rows = []
    # baselines
    for p in glob.glob("artifacts/baselines/*_results.json"):
        name = os.path.basename(p).replace("_results.json","")
        d = load_json(p)
        rows.append({"model": name, "split": "test", "macro_f1": d["test"]["macro_f1"]})
    # cnn
    p = "artifacts/cnn/kimcnn_results.json"
    if os.path.exists(p):
        d = load_json(p); rows.append({"model":"kimcnn","split":"test","macro_f1":d["test_macro_f1"]})
    # rnn
    p = "artifacts/rnn/bilstm_results.json"
    if os.path.exists(p):
        d = load_json(p); rows.append({"model":"bilstm_attn","split":"test","macro_f1":d["test_macro_f1"]})
    # transformer
    p = "artifacts/transformer/results.json"
    if os.path.exists(p):
        d = load_json(p); rows.append({"model":"distilbert+lora","split":"test","macro_f1":d["test_macro_f1"]})

    df = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
    print("\n=== Model ranking (macro-F1 on test) ===")
    print(df.to_string(index=False))

    out_md = "# Results\n\n| Model | Split | Macro-F1 |\n|---|---|---|\n" + \
             "\n".join([f"| {r['model']} | {r['split']} | {r['macro_f1']:.4f} |" for _, r in df.iterrows()])
    os.makedirs("artifacts", exist_ok=True)
    with open("artifacts/summary.md","w") as f: f.write(out_md)
    print("\nSaved: artifacts/summary.md")

if __name__ == "__main__":
    main()
