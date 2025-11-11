# 🏥 Healthcare Chatbot

A **stateless, safety-first healthcare chatbot** that predicts likely conditions from user-provided symptoms, provides concise medical explanations, and offers safe, evidence-based advice — powered by **AI + Groq LLM** for enhanced clarity and precision.

---

## ⚙️ Overview

| Component | Technology | Description |
|------------|-------------|-------------|
| **Backend** | Python (`TF-IDF + LogisticRegression`, Keras optional) | Core inference + condition classification |
| **LLM Enrichment** | [Groq API](https://groq.com/) | Enhances explanations with medically aligned, natural replies |
| **Frontend** | **Flask** (HTML + AJAX) | Lightweight and fast chatbot web interface |
| **Data** | CSV Knowledge Bases | `data/dataset.csv`, `data/symptom_description.csv`, `data/symptom_precaution.csv` |
| **Model Artifacts** | Pickle + TensorFlow | Stored under `/models` (`.pkl`, `.h5`, `.keras`) |
| **Core Modules** | `src/chatbot.py`, `src/inference.py`, `src/llm.py`, `src/utils.py` | Classification, rule-based reasoning, LLM enrichment |
| **Training** | `src/baseline.py` | Builds `system_metadata.pkl`, TF-IDF, and trained model |

---






## 🚀 Quick Start (Recommended)

### 1. Clone the repository

```bash
git clone https://github.com/riteshkatekar/Medical_Chatbot.git
cd Medical_Chatbot

```
### 2. Create a virtual environment and activate it
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

### 3. Install dependencies
Install all required libraries listed in requirements.txt.

```bash
pip install --upgrade pip && pip install -r requirements.txt

```


### 4. Set up your environment variables

Create a .env file in the root directory and add the following:


```bash
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL_ID=openai/gpt-oss-20b
LLM_TEMPERATURE=0.2
LLM_MAX_TOKENS=512


```

### 5. Build metadata & model (train baseline)
Run the training script to build the TF-IDF vectorizer and train the baseline Logistic Regression model. This generates necessary files in the models/ directory.

```bash
python src/baseline.py --data_path data/dataset.csv --desc_path data/symptom_description.csv --prec_path data/symptom_precaution.csv --save_dir models --tfidf_max_features 20000 --ngram_min 1 --ngram_max 2 --epochs 20 --batch_size 32 --synth_per_row 4
```
  
### 6. Run the interactive CLI (optional)
Test the core prediction engine via the command line.
```bash
python src/chatbot.py --models_dir models --interactive --threshold 0.05
```

### 7. Run the project
Launch the web interface. 
```bash
python app.py
```
Then open your browser and visit:

👉 http://127.0.0.1:8080/
