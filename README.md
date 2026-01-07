
# 🏥 AI-Powered Healthcare Text Intelligence 

An end-to-end **Healthcare-focused NLP & AI API** built using **Python, FastAPI, and modern ML techniques**.
This project demonstrates **sentiment analysis, keyword extraction, text summarization, semantic search,
and Retrieval-Augmented Generation (RAG)** using healthcare-related content.



---

## 🎯 Objective

To build an intelligent backend API that showcases:

- Healthcare NLP processing
- Transformer-based sentiment & summarization
- Embedding-based semantic search
- Retrieval-Augmented Generation (RAG) using healthcare knowledge
- Clean FastAPI architecture
- Deployment readiness using Docker

---

## 🏗️ Architecture Overview

Client → FastAPI → NLP Services → Embeddings → FAISS → RAG Context

All components run locally on CPU with no cloud dependency.

---

## ⚙️ Tech Stack

- Python 3.11
- FastAPI & Pydantic
- Hugging Face Transformers
- SpaCy
- SentenceTransformers
- FAISS
- Docker & Docker Compose

---

## 🚀 API Endpoints

### 1️⃣ Sentiment & Keyword Analysis
**POST /analyze**

Request:
```json
{ "text": "Regular health checkups make me feel safer." }
```

Response:
```json
{ "sentiment": "positive", "keywords": ["health", "checkups", "safer"] }
```

---

### 2️⃣ Text Summarization
**POST /summarize**

Request:
```json
{ "text": "Preventive healthcare includes routine exams and screenings..." }
```

Response:
```json
{ "summary": "Preventive healthcare helps detect diseases early." }
```

---

### 3️⃣ Semantic Search
**POST /semantic-search**

Request:
```json
{ "text": "Early disease prevention improves quality of life." }
```

Response:
```json
{ "query": "...", "results": ["Preventive healthcare focuses on disease prevention."] }
```

---

### 4️⃣ Healthcare RAG
**POST /rag-query**

Request:
```json
{ "query": "What is preventive healthcare?" }
```

Response:
```json
{
  "answer": "Preventive healthcare focuses on disease prevention and health maintenance.",
  "contexts": [
    "Preventive Healthcare",
    "Preventive healthcare focuses on disease prevention and health maintenance."
  ],
  "disclaimer": "This response is for informational purposes only and not medical advice."
}
```

---

## 🧠 Semantic Search vs RAG

Semantic Search finds meaning-based similar text.
RAG retrieves context and generates an informed answer.

---

## 🐳 Docker Setup

```bash
docker compose up --build
```

Swagger UI:
http://localhost:8000/docs

---

## 📁 Project Structure

- app/routes → API endpoints
- app/services → ML & RAG logic
- app/schemas → Request validation
- app/data → Healthcare knowledge base
- Dockerfile, docker-compose.yml, requirements.txt

---

## ⚠️ Disclaimer

This project is for educational purposes only and does not provide medical advice.

---

## 🎤  Summary

This project demonstrates a healthcare AI backend using FastAPI, embeddings, FAISS,
and RAG, fully containerized and CPU-friendly.

## Author
Mohsin Inamdar

## License
📄 License

MIT License

Copyright (c) 2026 Mohsin Inamdar

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
