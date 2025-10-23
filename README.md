# Sentiment Analysis on News Headlines  
**CNN vs RNN vs Transformer (with Strong Linear Baselines)**

> End-to-end comparison on `all-data.csv` using TF–IDF baselines, KimCNN, BiLSTM+Attention, and DistilBERT+LoRA. Trained on Apple Silicon (M1) with PyTorch MPS. This project was made after taking the Deep learning course exam to compare the results of each of the versions of the architectures taught in the course. 

---

## 1) Project Structure

.
├─ data/
│ └─ all-data.csv
├─ src/
│ ├─ data.py # load/clean dataset + stratified split
│ ├─ utils.py # save_json, plot_confusion
│ ├─ metrics.py # macro-F1, classification report, confusion matrix
│ ├─ models/
│ │ ├─ cnn.py # KimCNN
│ │ └─ rnn.py # BiLSTM + Attention
│ ├─ train_sklearn.py # TF–IDF + LogReg / LinearSVM
│ ├─ train_cnn.py # KimCNN training
│ ├─ train_rnn.py # BiLSTM+Attention training
│ └─ train_transformer.py# DistilBERT + LoRA (HF + PEFT)
├─ artifacts/
│ ├─ baselines/ # *_results.json, *_cm.png
│ ├─ cnn/ # kimcnn_results.json, kimcnn_cm.png, kimcnn.pt
│ ├─ rnn/ # bilstm_results.json, bilstm_cm.png, bilstm.pt
│ └─ transformer/ # checkpoints + results.json
├─ reports/
│ └─ README.md # this report
└─ requirements.txt

---

## 2) Dataset

- **File:** `data/all-data.csv` (loaded with `latin-1`, header quirk handled in `src/data.py`)
- **Task:** 3-way sentiment classification of short news headlines  
  Labels: `negative`, `neutral`, `positive`
- **Size:** ~4,845 examples (class-imbalanced; `negative` is the smallest class)
- **Split:** Stratified **70/15/15** (seed=42)
- **Preprocessing:** strip whitespace; lowercase labels; minimal text cleaning (tokenizers handle punctuation/case for models that need it)

---

## 3) Methods

### 3.1 Baselines (scikit-learn)
- **TF–IDF + Logistic Regression**  
  `ngram_range=(1,2)`, `max_features=100_000`, `class_weight="balanced"`, `max_iter=500`
- **TF–IDF + Linear SVM**  
  LinearSVC with `class_weight="balanced"`

> Rationale: strong, fast baselines for short texts; hard to beat on headlines.

### 3.2 CNN
- **KimCNN**  
  Kernels **(3,4,5)**, `n_filters=100`, `emb_dim=200`, dropout=0.5  
  Max sequence length 64; AdamW with lr=1e-3.

### 3.3 RNN
- **BiLSTM + Attention**  
  Single-layer bidirectional LSTM (`hidden=256`), simple attention head, `emb_dim=200`, dropout=0.3  
  Max length 64; AdamW with lr=1e-3.

### 3.4 Transformer
- **DistilBERT + LoRA (PEFT)**  
  Tokenizer max length **96**, batch **16**, **3 epochs**  
  LoRA: rank `r=8`, `alpha=16`, dropout=0.05 on attention projection modules  
  HF `Trainer` (metric: macro-F1 on validation; load best at end)

---

## 4) Training Environment

- **Hardware:** Apple Silicon (M1)  
- **Frameworks:** PyTorch (MPS backend), scikit-learn, Hugging Face Transformers + PEFT  
- **How to run:** from project root using `python -m ...` so relative imports resolve.

---

## 5) Reproduction

# Baselines
'python -m src.train_sklearn --model logreg'
'python -m src.train_sklearn --model linearsvm'

# CNN
'python -m src.train_cnn --epochs 8 --batch_size 32 --max_len 64'

# RNN
'python -m src.train_rnn --epochs 8 --batch_size 32 --max_len 64'

# Transformer (LoRA)
'python -m src.train_transformer --model_name distilbert-base-uncased \ --use_lora 1 --epochs 3 --batch_size 16 --max_len 96'

# Aggregate summary
'python -m src.analyze'
---
## 6) Results

Primary metric: macro-F1 on the test split (robust to class imbalance).

Aggregated ranking (from artifacts/summary.md):
| Model | Split | Macro-F1 |
|---|---|---|
| logreg | test | 0.7167 |
| linearsvm | test | 0.7090 |
| bilstm_attn | test | 0.7049 |
| kimcnn | test | 0.7046 |
| distilbert+lora | test | 0.5628 |

Artifacts you can consult:

- Baselines:artifacts/baselines/logreg_results.json, artifacts/baselines/logreg_test_cm.png, artifacts/baselines linearsvm_results.json, artifacts/baselines/linearsvm_test_cm.png

- CNN: artifacts/cnn/kimcnn_results.json, artifacts/cnn/kimcnn_cm.png

- RNN: artifacts/rnn/bilstm_results.json, artifacts/rnn/bilstm_cm.png

- Transformer:artifacts/transformer/results.json and artifacts/transformer/checkpoint-*
## 7) Analysis
### 7.1 What worked best and why

TF–IDF + Logistic Regression achieved the top macro-F1. On short, information-dense headlines, word/character n-grams capture sentiment cues (negations, intensifiers, finance verbs) extremely well.

Linear SVM was close behind, which is typical for sparse high-dimensional text features.

KimCNN and BiLSTM + Attention were competitive but did not surpass the TF–IDF baselines. With short sequences, local n-grams plus simple pooling often match or beat sequence models unless heavily tuned or pre-initialized with strong embeddings.

DistilBERT + LoRA underperformed in this run. On small, short-text datasets, adapter fine-tuning with conservative schedules can trail well-tuned classical baselines.

### 7.2 Likely reasons the transformer trailed

Epoch budget: only 3 epochs; small datasets often benefit from 5–8 epochs with early stopping.

Learning rate: 2e-5 is common for full fine-tuning; LoRA layers often prefer 1e-4 to 2e-4.

Adapter coverage: changing only attention projections can be limiting; adding FFN adapters (or IA³) helps.

Class imbalance: no explicit class weighting/oversampling in the HF run.

Token/batch constraints: conservative max_length and batch sizes reduce adaptation headroom (even if headlines are short).

### 7.3 Recommendations (next experiments)

Transformer

Try smaller, fast models such as MiniLM (nreimers/MiniLM-L6-H384-uncased) or ALBERT-base.

Train 5–8 epochs, add 10% warmup, weight decay = 0.01.

Increase adapter LR to ~1e-4; keep batch 8–16; use gradient accumulation if needed.

Add class weights (custom compute_loss) or oversample the minority class.

Sanity check a frozen encoder + linear head baseline.

Neural (CNN/RNN)

Initialize with pretrained fastText/GloVe word vectors.

Consider focal loss for the minority class; tune dropout and filter/hidden sizes.

Classical

Tune TF–IDF (add character n-grams), and try Calibrated SVM if you need probabilities.

## 8) Error Analysis
What to inspect

Confident misclassifications: review the most confident wrong predictions to spot patterns such as negations (“not”, “no longer”), hedging (“may”, “plans to”), and finance-specific jargon.

Confusion matrices (see artifacts/*/*_cm.png): identify which classes are most often confused (commonly negative vs. neutral).

Reporting ideas

Include 5–10 misclassified examples with gold label, predicted label, and a one-line hypothesis.

If available, add token highlights via attention visualizations or LIME/SHAP for 2–3 examples.

## 9) Limitations

Dataset size is modest; results may not directly transfer to long-form news or other domains.

Limited hyperparameter search to keep training M1-friendly; deeper tuning likely lifts neural/transformer scores.

Single stratified split reported; no k-fold averages (mean ± std) in this summary.

## 10) Acknowledgments

Libraries: scikit-learn, PyTorch (MPS), Hugging Face Transformers, PEFT.

Architectures: KimCNN, BiLSTM with attention, DistilBERT with LoRA adapters.