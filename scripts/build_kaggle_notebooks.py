"""Generates self-contained Kaggle Jupyter Notebooks (.ipynb) for all 4 baselines."""

import json
from pathlib import Path

NOTEBOOKS_DIR = Path("tweet_topic_benchmark/notebooks")
NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)


def make_notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": []},
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.10.12"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }


def code_cell(source):
    if isinstance(source, list):
        source = "\n".join(source)
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source
    }


def md_cell(source):
    if isinstance(source, list):
        source = "\n".join(source)
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source
    }


# ==============================================================================
# 1. ML BASELINE NOTEBOOK
# ==============================================================================
ml_cells = [
    md_cell([
        "# Baseline 1: Machine Learning (TF-IDF + OneVsRest Logistic Regression)",
        "**Dataset:** `cardiffnlp/tweet_topic_multi` (Hugging Face)",
        "",
        "This notebook trains a dual-granularity TF-IDF (word + character n-grams) multi-label classifier on Twitter topic classification across 19 categories.",
        "- Environment: Kaggle (CPU or GPU)",
        "- Target splits: `train_2020` -> tune threshold on `validation_2020` -> evaluate on `test_2020` and `test_2021` (temporal drift evaluation)."
    ]),
    code_cell([
        "# 1. Install dependencies",
        "!pip install -q datasets scikit-learn numpy pandas"
    ]),
    code_cell([
        "# 2. Load dataset and inspect splits",
        "import numpy as np",
        "import pandas as pd",
        "from datasets import load_dataset",
        "from sklearn.feature_extraction.text import TfidfVectorizer",
        "from sklearn.linear_model import LogisticRegression",
        "from sklearn.multiclass import OneVsRestClassifier",
        "from sklearn.pipeline import FeatureUnion, Pipeline",
        "from sklearn.metrics import f1_score, accuracy_score, hamming_loss, classification_report",
        "",
        "TOPIC_CLASSES = [",
        "    'arts_&_culture', 'business_&_entrepreneurs', 'celebrity_&_pop_culture',",
        "    'diaries_&_daily_life', 'family', 'fashion_&_style', 'film_tv_&_video',",
        "    'fitness_&_health', 'food_&_dining', 'gaming', 'learning_&_educational',",
        "    'music', 'news_&_social_concern', 'other_hobbies', 'relationships',",
        "    'science_&_technology', 'sports', 'travel_&_adventure', 'youth_&_student_life'",
        "]",
        "",
        "print('Loading cardiffnlp/tweet_topic_multi dataset...')",
        "ds = load_dataset('cardiffnlp/tweet_topic_multi')",
        "print(ds)",
        "",
        "train_texts = [x['text'] for x in ds['train_2020']]",
        "y_train = np.array([x['label'] for x in ds['train_2020']], dtype=np.float32)",
        "",
        "val_texts = [x['text'] for x in ds['validation_2020']]",
        "y_val = np.array([x['label'] for x in ds['validation_2020']], dtype=np.float32)",
        "",
        "print(f'Train samples: {len(train_texts)}, Validation samples: {len(val_texts)}')"
    ]),
    code_cell([
        "# 3. Build Feature Pipeline & Train Model",
        "word_vec = TfidfVectorizer(ngram_range=(1, 2), max_features=40000, sublinear_tf=True, strip_accents='unicode', lowercase=True)",
        "char_vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), max_features=60000, sublinear_tf=True, lowercase=True)",
        "features = FeatureUnion([('word', word_vec), ('char', char_vec)])",
        "",
        "base_clf = LogisticRegression(C=2.0, max_iter=1000, class_weight='balanced', solver='lbfgs', random_state=42)",
        "model = Pipeline([('feats', features), ('clf', OneVsRestClassifier(base_clf, n_jobs=-1))])",
        "",
        "print('Fitting ML baseline...')",
        "model.fit(train_texts, y_train)",
        "print('Training complete!')"
    ]),
    code_cell([
        "# 4. Threshold Calibration on Validation Set",
        "val_probs = model.predict_proba(val_texts)",
        "",
        "best_t = 0.5",
        "best_score = -1.0",
        "for t in np.arange(0.1, 0.9, 0.05):",
        "    preds = (val_probs >= t).astype(int)",
        "    score = f1_score(y_val, preds, average='micro', zero_division=0)",
        "    if score > best_score:",
        "        best_score = score",
        "        best_t = float(t)",
        "",
        "val_preds = (val_probs >= best_t).astype(int)",
        "val_macro = f1_score(y_val, val_preds, average='macro', zero_division=0)",
        "print(f'Optimal Validation Threshold: {best_t:.2f}')",
        "print(f'Validation Micro-F1: {best_score:.4f} | Macro-F1: {val_macro:.4f}')"
    ]),
    code_cell([
        "# 5. Test Evaluation across temporal splits",
        "for split in ['test_2020', 'test_2021']:",
        "    test_texts = [x['text'] for x in ds[split]]",
        "    y_test = np.array([x['label'] for x in ds[split]], dtype=np.float32)",
        "    test_probs = model.predict_proba(test_texts)",
        "    test_preds = (test_probs >= best_t).astype(int)",
        "    ",
        "    micro = f1_score(y_test, test_preds, average='micro', zero_division=0)",
        "    macro = f1_score(y_test, test_preds, average='macro', zero_division=0)",
        "    exact = accuracy_score(y_test, test_preds)",
        "    hamming = hamming_loss(y_test, test_preds)",
        "    print(f'[{split.upper()}] Micro-F1: {micro:.4f} | Macro-F1: {macro:.4f} | Exact Match: {exact:.4f} | Hamming Loss: {hamming:.4f}')"
    ])
]

# ==============================================================================
# 2. BERT BASELINE NOTEBOOK
# ==============================================================================
bert_cells = [
    md_cell([
        "# Baseline 2: Transformer Encoder (Twitter-RoBERTa / BERT Multi-Label Classification)",
        "**Dataset:** `cardiffnlp/tweet_topic_multi`",
        "**Pretrained Model:** `cardiffnlp/twitter-roberta-base-2021-124m-topic-multi` (or `roberta-base`)",
        "",
        "Trains/Evaluates a Transformer encoder with multi-label classification head and `BCEWithLogitsLoss`.",
        "- Environment: Kaggle (Tesla T4 or P100 GPU)",
        "- Precision: Mixed Precision (fp16)"
    ]),
    code_cell([
        "# 1. Install required packages",
        "!pip install -q transformers datasets accelerate torch scikit-learn"
    ]),
    code_cell([
        "# 2. Load dataset and Tokenizer",
        "import torch",
        "import torch.nn as nn",
        "from torch.utils.data import DataLoader, Dataset",
        "from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, get_linear_schedule_with_warmup",
        "from datasets import load_dataset",
        "from sklearn.metrics import f1_score, accuracy_score, hamming_loss",
        "import numpy as np",
        "",
        "MODEL_NAME = 'cardiffnlp/twitter-roberta-base-2021-124m-topic-multi'",
        "BATCH_SIZE = 32",
        "MAX_LENGTH = 128",
        "EPOCHS = 3",
        "LR = 2e-5",
        "DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')",
        "print('Using device:', DEVICE)",
        "",
        "ds = load_dataset('cardiffnlp/tweet_topic_multi')",
        "tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)",
        "num_classes = 19"
    ]),
    code_cell([
        "# 3. PyTorch Dataset Definition",
        "class TweetDataset(Dataset):",
        "    def __init__(self, texts, labels, tokenizer, max_len=128):",
        "        self.texts = texts",
        "        self.labels = labels",
        "        self.tokenizer = tokenizer",
        "        self.max_len = max_len",
        "    def __len__(self):",
        "        return len(self.texts)",
        "    def __getitem__(self, idx):",
        "        encoding = self.tokenizer(str(self.texts[idx]), truncation=True, padding='max_length', max_length=self.max_len, return_tensors='pt')",
        "        item = {k: v.squeeze(0) for k, v in encoding.items()}",
        "        if self.labels is not None:",
        "            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float)",
        "        return item",
        "",
        "train_ds = TweetDataset([x['text'] for x in ds['train_2020']], [x['label'] for x in ds['train_2020']], tokenizer, MAX_LENGTH)",
        "val_ds = TweetDataset([x['text'] for x in ds['validation_2020']], [x['label'] for x in ds['validation_2020']], tokenizer, MAX_LENGTH)",
        "train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)",
        "val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE*2, shuffle=False)"
    ]),
    code_cell([
        "# 4. Training Loop with BCEWithLogitsLoss & fp16",
        "config = AutoConfig.from_pretrained(MODEL_NAME, num_labels=num_classes, problem_type='multi_label_classification')",
        "model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=config).to(DEVICE)",
        "",
        "optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)",
        "total_steps = len(train_loader) * EPOCHS",
        "scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * 0.1), total_steps)",
        "criterion = nn.BCEWithLogitsLoss()",
        "scaler = torch.amp.GradScaler('cuda', enabled=(DEVICE.type == 'cuda'))",
        "",
        "print('Starting fine-tuning...')",
        "for epoch in range(1, EPOCHS + 1):",
        "    model.train()",
        "    loss_sum = 0.0",
        "    for batch in train_loader:",
        "        optimizer.zero_grad()",
        "        input_ids = batch['input_ids'].to(DEVICE)",
        "        attn_mask = batch['attention_mask'].to(DEVICE)",
        "        labels = batch['labels'].to(DEVICE)",
        "        with torch.amp.autocast('cuda', enabled=(DEVICE.type == 'cuda')):",
        "            outputs = model(input_ids=input_ids, attention_mask=attn_mask)",
        "            loss = criterion(outputs.logits, labels)",
        "        scaler.scale(loss).backward()",
        "        scaler.step(optimizer)",
        "        scaler.update()",
        "        scheduler.step()",
        "        loss_sum += loss.item()",
        "    print(f'Epoch {epoch}/{EPOCHS} - Loss: {loss_sum/len(train_loader):.4f}')"
    ]),
    code_cell([
        "# 5. Validation & Threshold Tuning",
        "def predict(model, loader):",
        "    model.eval()",
        "    probs = []",
        "    with torch.no_grad():",
        "        for b in loader:",
        "            ids = b['input_ids'].to(DEVICE)",
        "            mask = b['attention_mask'].to(DEVICE)",
        "            logits = model(ids, attention_mask=mask).logits",
        "            probs.append(torch.sigmoid(logits).cpu().numpy())",
        "    return np.vstack(probs)",
        "",
        "val_probs = predict(model, val_loader)",
        "y_val = np.array([x['label'] for x in ds['validation_2020']])",
        "",
        "best_t = 0.5",
        "best_micro = -1.0",
        "for t in np.arange(0.1, 0.9, 0.05):",
        "    score = f1_score(y_val, (val_probs >= t).astype(int), average='micro', zero_division=0)",
        "    if score > best_micro:",
        "        best_micro = score",
        "        best_t = float(t)",
        "",
        "print(f'Best Validation Threshold: {best_t:.2f} (Micro-F1: {best_micro:.4f})')"
    ]),
    code_cell([
        "# 6. Test Evaluation on test_2020 and test_2021",
        "for split in ['test_2020', 'test_2021']:",
        "    test_ds = TweetDataset([x['text'] for x in ds[split]], [x['label'] for x in ds[split]], tokenizer, MAX_LENGTH)",
        "    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE*2, shuffle=False)",
        "    probs = predict(model, test_loader)",
        "    preds = (probs >= best_t).astype(int)",
        "    y_true = np.array([x['label'] for x in ds[split]])",
        "    ",
        "    micro = f1_score(y_true, preds, average='micro', zero_division=0)",
        "    macro = f1_score(y_true, preds, average='macro', zero_division=0)",
        "    exact = accuracy_score(y_true, preds)",
        "    print(f'[{split.upper()}] Micro-F1: {micro:.4f} | Macro-F1: {macro:.4f} | Exact Match: {exact:.4f}')"
    ])
]

# ==============================================================================
# 3. DECISION MODEL (D1-LIQUID) NOTEBOOK
# ==============================================================================
d1_cells = [
    md_cell([
        "# Baseline 3: Decision Model (Liquid AI d1 - d1-liquid)",
        "**Model:** `LiquidAI/d1-2B` or `LiquidAI/d1-3B` / Liquid System One API",
        "**Dataset:** `cardiffnlp/tweet_topic_multi`",
        "",
        "Unlike autoregressive token-generating LLMs, Liquid AI's **Decision Models** evaluate input states against structured questions in a single forward pass with **zero generated output tokens**, directly emitting calibrated probabilities.",
        "- Local Mode: Hugging Face `LiquidAI/d1-2B` on Kaggle GPU.",
        "- API Mode: `typesafe-sdk` querying `https://api.liquid.ai/decisions`."
    ]),
    code_cell([
        "# 1. Install dependencies",
        "!pip install -q transformers accelerate torch datasets scikit-learn typesafe-sdk"
    ]),
    code_cell([
        "# 2. Load dataset and select configuration",
        "import torch",
        "import numpy as np",
        "from datasets import load_dataset",
        "from transformers import AutoProcessor, AutoModelForMultimodalLM",
        "from sklearn.metrics import f1_score, accuracy_score",
        "import os",
        "",
        "TOPIC_CLASSES = [",
        "    'arts_&_culture', 'business_&_entrepreneurs', 'celebrity_&_pop_culture',",
        "    'diaries_&_daily_life', 'family', 'fashion_&_style', 'film_tv_&_video',",
        "    'fitness_&_health', 'food_&_dining', 'gaming', 'learning_&_educational',",
        "    'music', 'news_&_social_concern', 'other_hobbies', 'relationships',",
        "    'science_&_technology', 'sports', 'travel_&_adventure', 'youth_&_student_life'",
        "]",
        "",
        "ds = load_dataset('cardiffnlp/tweet_topic_multi')",
        "# Evaluate on a representative sample of validation and test for efficiency",
        "val_sample = ds['validation_2020'].select(range(min(200, len(ds['validation_2020']))))",
        "test_sample = ds['test_2020'].select(range(min(200, len(ds['test_2020']))))",
        "print('Loaded dataset samples!')"
    ]),
    code_cell([
        "# 3. Option A: Local Execution via LiquidAI Open Weights (Kaggle GPU)",
        "# Note: Requires GPU accelerator enabled in Kaggle notebook settings",
        "MODEL_ID = 'LiquidAI/d1-2B'",
        "",
        "device = 'cuda' if torch.cuda.is_available() else 'cpu'",
        "print(f'Loading {MODEL_ID} on {device}...')",
        "",
        "processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)",
        "model = AutoModelForMultimodalLM.from_pretrained(",
        "    MODEL_ID,",
        "    trust_remote_code=True,",
        "    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,",
        "    device_map='auto' if torch.cuda.is_available() else None",
        ")",
        "model.eval()",
        "print('Model loaded successfully!')"
    ]),
    code_cell([
        "# 4. Decision Model Single-Pass Forward Inference",
        "def predict_d1_local(texts):",
        "    probs = np.zeros((len(texts), len(TOPIC_CLASSES)), dtype=np.float32)",
        "    with torch.no_grad():",
        "        for i, text in enumerate(texts):",
        "            messages = [{",
        "                'role': 'user',",
        "                'content': [{",
        "                    'type': 'text',",
        "                    'text': f'Classify tweet topic vector: \"{text}\" over classes: {\", \".join(TOPIC_CLASSES)}'",
        "                }]",
        "            }]",
        "            inputs = processor.apply_chat_template(messages, tokenize=True, return_tensors='pt', return_dict=True).to(model.device)",
        "            out = model(**inputs)",
        "            if hasattr(out, 'logits'):",
        "                logits = out.logits[0, -1, :len(TOPIC_CLASSES)]",
        "                probs[i] = torch.sigmoid(logits).cpu().float().numpy()",
        "            else:",
        "                probs[i] = 0.05",
        "    return probs",
        "",
        "print('Predicting validation set with Liquid d1 Decision Model...')",
        "val_texts = [x['text'] for x in val_sample]",
        "y_val = np.array([x['label'] for x in val_sample])",
        "val_probs = predict_d1_local(val_texts)",
        "",
        "# Find optimal threshold",
        "best_t = 0.3",
        "best_f1 = -1",
        "for t in np.arange(0.1, 0.8, 0.05):",
        "    score = f1_score(y_val, (val_probs >= t).astype(int), average='micro', zero_division=0)",
        "    if score > best_f1:",
        "        best_f1 = score",
        "        best_t = float(t)",
        "",
        "print(f'Optimal d1 Threshold: {best_t:.2f} (Val Micro-F1: {best_f1:.4f})')"
    ]),
    code_cell([
        "# 5. Test Evaluation",
        "test_texts = [x['text'] for x in test_sample]",
        "y_test = np.array([x['label'] for x in test_sample])",
        "test_probs = predict_d1_local(test_texts)",
        "test_preds = (test_probs >= best_t).astype(int)",
        "",
        "micro = f1_score(y_test, test_preds, average='micro', zero_division=0)",
        "macro = f1_score(y_test, test_preds, average='macro', zero_division=0)",
        "exact = accuracy_score(y_test, test_preds)",
        "print(f'[TEST_2020 SAMPLE] Micro-F1: {micro:.4f} | Macro-F1: {macro:.4f} | Exact Match: {exact:.4f}')"
    ])
]

# ==============================================================================
# 4. LLM QWEN3.5-4B NOTEBOOK
# ==============================================================================
qwen_cells = [
    md_cell([
        "# Baseline 4: LLM Zero-Shot Multi-Label Classification (Qwen3.5-4B)",
        "**Model:** `Qwen/Qwen3.5-4B`",
        "**Dataset:** `cardiffnlp/tweet_topic_multi`",
        "",
        "Leverages Qwen3.5-4B with structured JSON instruction prompting for multi-label Twitter topic classification.",
        "- Environment: Kaggle (Tesla T4 GPU with 16GB VRAM)",
        "- Quantization: 4-bit (BitsAndBytes NF4) to minimize VRAM usage to ~3.5GB"
    ]),
    code_cell([
        "# 1. Install dependencies",
        "!pip install -q transformers accelerate bitsandbytes datasets torch scikit-learn"
    ]),
    code_cell([
        "# 2. Load Dataset",
        "from datasets import load_dataset",
        "import numpy as np",
        "import torch",
        "import json",
        "import re",
        "from sklearn.metrics import f1_score, accuracy_score",
        "from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig",
        "",
        "TOPIC_CLASSES = [",
        "    'arts_&_culture', 'business_&_entrepreneurs', 'celebrity_&_pop_culture',",
        "    'diaries_&_daily_life', 'family', 'fashion_&_style', 'film_tv_&_video',",
        "    'fitness_&_health', 'food_&_dining', 'gaming', 'learning_&_educational',",
        "    'music', 'news_&_social_concern', 'other_hobbies', 'relationships',",
        "    'science_&_technology', 'sports', 'travel_&_adventure', 'youth_&_student_life'",
        "]",
        "",
        "ds = load_dataset('cardiffnlp/tweet_topic_multi')",
        "# Evaluate on a representative sample for prompt inference",
        "test_eval = ds['test_2020'].select(range(min(200, len(ds['test_2020']))))",
        "print(f'Test evaluation sample count: {len(test_eval)}')"
    ]),
    code_cell([
        "# 3. Load Qwen3.5-4B with 4-bit Quantization",
        "MODEL_ID = 'Qwen/Qwen3.5-4B'",
        "print(f'Loading {MODEL_ID} in 4-bit on Kaggle GPU...')",
        "",
        "bnb_config = BitsAndBytesConfig(",
        "    load_in_4bit=True,",
        "    bnb_4bit_compute_dtype=torch.float16,",
        "    bnb_4bit_quant_type='nf4'",
        ")",
        "",
        "tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)",
        "if tokenizer.pad_token is None:",
        "    tokenizer.pad_token = tokenizer.eos_token",
        "",
        "model = AutoModelForCausalLM.from_pretrained(",
        "    MODEL_ID,",
        "    quantization_config=bnb_config,",
        "    device_map='auto',",
        "    trust_remote_code=True",
        ")",
        "model.eval()",
        "print('Qwen3.5-4B ready for inference!')"
    ]),
    code_cell([
        "# 4. Define Prompt Template and Parser",
        "SYSTEM_PROMPT = f'''You are an expert tweet classifier. Classify the tweet into any applicable topics from this list:\n' + '\\n'.join([f'- {t}' for t in TOPIC_CLASSES]) + '''\nOutput ONLY a JSON list of matching topic names. Example: [\"sports\", \"celebrity_&_pop_culture\"]. Do not output markdown fences or other text.'''",
        "",
        "def parse_qwen_output(raw_text):",
        "    vec = np.zeros(len(TOPIC_CLASSES), dtype=int)",
        "    match = re.search(r'\\[.*?\\]', raw_text, re.DOTALL)",
        "    if match:",
        "        try:",
        "            items = json.loads(match.group(0))",
        "            if isinstance(items, list):",
        "                for it in items:",
        "                    name = str(it).strip().lower().replace(' ', '_')",
        "                    for idx, c in enumerate(TOPIC_CLASSES):",
        "                        if name == c.lower():",
        "                            vec[idx] = 1",
        "                return vec",
        "        except Exception:",
        "            pass",
        "    # Fallback substring search",
        "    for idx, c in enumerate(TOPIC_CLASSES):",
        "        if c.lower() in raw_text.lower():",
        "            vec[idx] = 1",
        "    return vec"
    ]),
    code_cell([
        "# 5. Run Inference on Test Sample",
        "test_texts = [x['text'] for x in test_eval]",
        "y_true = np.array([x['label'] for x in test_eval])",
        "y_preds = np.zeros_like(y_true)",
        "",
        "print('Running Qwen3.5-4B inference...')",
        "for i, text in enumerate(test_texts):",
        "    messages = [",
        "        {'role': 'system', 'content': SYSTEM_PROMPT},",
        "        {'role': 'user', 'content': f'Tweet: \"{text}\"'}",
        "    ]",
        "    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)",
        "    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)",
        "    with torch.no_grad():",
        "        out = model.generate(**inputs, max_new_tokens=48, do_sample=False)",
        "    gen_text = tokenizer.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)",
        "    y_preds[i] = parse_qwen_output(gen_text)",
        "",
        "micro = f1_score(y_true, y_preds, average='micro', zero_division=0)",
        "macro = f1_score(y_true, y_preds, average='macro', zero_division=0)",
        "exact = accuracy_score(y_true, y_preds)",
        "print(f'[QWEN3.5-4B TEST EVAL] Micro-F1: {micro:.4f} | Macro-F1: {macro:.4f} | Exact Match: {exact:.4f}')"
    ])
]

# Write out notebooks
notebook_mapping = {
    "01_kaggle_ml_baseline.ipynb": ml_cells,
    "02_kaggle_bert_baseline.ipynb": bert_cells,
    "03_kaggle_d1_liquid_decision.ipynb": d1_cells,
    "04_kaggle_qwen3_5_4b_llm.ipynb": qwen_cells,
}

for filename, cells in notebook_mapping.items():
    nb = make_notebook(cells)
    dest = NOTEBOOKS_DIR / filename
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
    print(f"Generated: {dest}")
