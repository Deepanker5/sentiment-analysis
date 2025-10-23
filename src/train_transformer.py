# src/train_transformer.py
import argparse, os, numpy as np, json
from datasets import Dataset, DatasetDict
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer)
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from src.data import load_dataset, split_stratified, LABEL_ORDER

def build_hf_datasets(seed=42):
    df = load_dataset()
    tr, va, te = split_stratified(df, seed=seed)
    def map_labels(df_):
        return {"text": df_["text"].tolist(), "labels": [LABEL_ORDER.index(x) for x in df_["label"]]}
    return DatasetDict({
        "train": Dataset.from_dict(map_labels(tr)),
        "val":   Dataset.from_dict(map_labels(va)),
        "test":  Dataset.from_dict(map_labels(te)),
    })

def run(model_name="distilbert-base-uncased", use_lora=True, epochs=3, batch_size=16, max_len=96, lr=2e-5, seed=42):
    ds = build_hf_datasets(seed)
    tok = AutoTokenizer.from_pretrained(model_name)
    def tokenize(batch): return tok(batch["text"], truncation=True, padding="max_length", max_length=max_len)
    ds = ds.map(tokenize, batched=True)
    ds.set_format(type="torch", columns=["input_ids","attention_mask","labels"])

    base = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
    if use_lora:
        config = LoraConfig(task_type=TaskType.SEQ_CLS, r=8, lora_alpha=16, lora_dropout=0.05,
                            target_modules=["q_lin","k_lin","v_lin","out_lin"])  # DistilBERT modules
        model = get_peft_model(base, config)
    else:
        model = base

    args = TrainingArguments(
        output_dir="artifacts/transformer",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        load_best_model_at_end=True,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        learning_rate=lr,
        metric_for_best_model="f1",
        greater_is_better=True,
        seed=seed,
        report_to="none"
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {"f1": f1_score(labels, preds, average="macro")}

    trainer = Trainer(
        model=model, args=args,
        train_dataset=ds["train"], eval_dataset=ds["val"],
        tokenizer=tok, compute_metrics=compute_metrics
    )

    trainer.train()
    test_out = trainer.predict(ds["test"])
    preds = np.argmax(test_out.predictions, axis=-1)
    y = test_out.label_ids
    f1 = f1_score(y, preds, average="macro")
    rep = classification_report(y, preds, target_names=LABEL_ORDER, digits=4)
    cm = confusion_matrix(y, preds).tolist()

    os.makedirs("artifacts/transformer", exist_ok=True)
    with open("artifacts/transformer/results.json","w") as f:
        json.dump({"test_macro_f1": float(f1), "report": rep, "cm": cm}, f, indent=2)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="distilbert-base-uncased")
    ap.add_argument("--use_lora", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_len", type=int, default=96)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    run(args.model_name, bool(args.use_lora), args.epochs, args.batch_size, args.max_len, args.lr, args.seed)
