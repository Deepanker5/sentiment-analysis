# src/train_rnn.py
import argparse, os, re, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from src.data import load_dataset, split_stratified, label_to_id, LABEL_ORDER
from src.metrics import macro_f1, full_report, confmat
from src.utils import save_json, plot_confusion
from src.models.rnn import BiLSTMAttn

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

def tokenize(t): 
    return re.findall(r"\w+|\S", t.lower())

def build_vocab(texts, min_freq=2, max_size=30000):
    cnt = Counter()
    for t in texts: cnt.update(tokenize(t))
    vocab = {"<pad>":0, "<unk>":1}
    for tok, n in cnt.most_common():
        if n < min_freq: break
        if tok not in vocab: vocab[tok] = len(vocab)
        if len(vocab) >= max_size: break
    return vocab

def numericalize(text, vocab, max_len=64):
    ids = [vocab.get(tok, 1) for tok in tokenize(text)][:max_len]
    return ids + [0]*(max_len - len(ids))

class TxtDS(Dataset):
    def __init__(self, df, vocab, max_len=64):
        self.x = [torch.tensor(numericalize(t, vocab, max_len)) for t in df["text"]]
        self.y = [label_to_id(l) for l in df["label"]]
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.x[i], self.y[i]

def run(epochs=8, batch_size=32, lr=1e-3, emb_dim=200, hidden=256, max_len=64, seed=42):
    torch.manual_seed(seed)
    df = load_dataset()
    tr, va, te = split_stratified(df, seed=seed)
    vocab = build_vocab(tr["text"])
    train_ds, val_ds, test_ds = TxtDS(tr, vocab, max_len), TxtDS(va, vocab, max_len), TxtDS(te, vocab, max_len)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl   = DataLoader(val_ds, batch_size=batch_size)
    test_dl  = DataLoader(test_ds, batch_size=batch_size)

    model = BiLSTMAttn(len(vocab), emb_dim, hidden, num_classes=3).to(DEVICE)
    opt, crit = torch.optim.AdamW(model.parameters(), lr=lr), nn.CrossEntropyLoss()

    best_f1, best_state = -1, None
    for ep in range(1, epochs+1):
        model.train()
        for xb, yb in train_dl:
            xb, yb = xb.to(DEVICE), torch.tensor(yb).to(DEVICE)
            opt.zero_grad(); loss = crit(model(xb), yb); loss.backward(); opt.step()
        # validate
        model.eval(); preds, gold = [], []
        with torch.no_grad():
            for xb, yb in val_dl:
                xb = xb.to(DEVICE)
                logits = model(xb).cpu()
                preds += logits.argmax(-1).tolist()
                gold  += yb
        f1 = macro_f1(gold, preds)
        if f1 > best_f1: best_f1, best_state = f1, {k:v.cpu() for k,v in model.state_dict().items()}

    # test
    model.load_state_dict(best_state)
    model.eval(); preds, gold = [], []
    with torch.no_grad():
        for xb, yb in test_dl:
            xb = xb.to(DEVICE)
            logits = model(xb).cpu()
            preds += logits.argmax(-1).tolist()
            gold  += yb

    f1 = macro_f1(gold, preds); rep = full_report(gold, preds, LABEL_ORDER)
    cm = confmat(gold, preds, LABEL_ORDER)
    os.makedirs("artifacts/rnn", exist_ok=True)
    torch.save({"state_dict": best_state, "vocab": vocab}, "artifacts/rnn/bilstm.pt")
    plot_confusion(cm, LABEL_ORDER, "artifacts/rnn/bilstm_cm.png")
    save_json({"test_macro_f1": float(f1), "report": rep, "cm": cm.tolist()}, "artifacts/rnn/bilstm_results.json")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--emb_dim", type=int, default=200)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--max_len", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()
    run(**vars(args))
