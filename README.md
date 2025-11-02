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
