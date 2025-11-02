# 🏥 Healthcare Chatbot

A **stateless, safety-first healthcare chatbot** that predicts likely conditions from user-provided symptoms, explains reasoning, and lists safe precautions.

---

## ⚙️ Overview

- **Backend:** Python (`TF-IDF + LogisticRegression` baseline; rule/fuzzy-based `InferenceEngine`)
- **Frontend:** Streamlit (web UI) with embedded **ChatGPT-style mic** using the browser **Web Speech API**
- **Data:** CSV knowledge bases (`data/dataset.csv`, `data/symptom_description.csv`, `data/symptom_precaution.csv`)
- **CLI:** `src/chatbot.py` for interactive/single query usage
- **Training:** `src/baseline.py` creates `models/system_metadata.pkl`, TF-IDF and sklearn model

---



## 🚀 Quick Start (Recommended)

### 1. Clone the repository

```bash
git clone https://github.com/riteshkatekar/Medical_Chatbot.git
cd Medical_Chatbot

```
2. Create a virtual environment and activate it
A virtual environment ensures project dependencies are isolated.

## Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## Windows (PowerShell)
```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1

```

3. Install dependencies
Install all required libraries listed in requirements.txt.

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

4. Ensure your data CSVs are in the data/ folder
Verify the following files exist to be used as the knowledge base:

data/dataset.csv
data/symptom_description.csv
data/symptom_precaution.csv

---

5. Build metadata & model (train baseline)
Run the training script to build the TF-IDF vectorizer and train the baseline Logistic Regression model. This generates necessary files in the models/ directory.

```bash
python src/baseline.py --data_path data/dataset.csv --desc_path data/symptom_description.csv --prec_path data/symptom_precaution.csv --save_dir models/ --tfidf_max_features 20000 --ngram_min 1 --ngram_max 2
```
  
6. Run the interactive CLI (optional)
Test the core prediction engine via the command line.
```bash
python src/chatbot.py --models_dir models --interactive --threshold 0.2
```

8. Run the Streamlit UI (open browser when started)
Launch the web interface. Streamlit will automatically open a tab in your default browser.

```bash

streamlit run app.py
```
