# ⛏️ Mining Knowledge AI Assistant (RAG + LLM)

An AI-powered document question answering system built using **Retrieval-Augmented Generation (RAG)** and **Streamlit**.
The system enables users to upload mining engineering documents (PDF format), extract text chunks, generate vector embeddings, and interactively ask questions with context-backed AI answers.

---

## 🌟 Key Features

* **Interactive Streamlit Web Interface**: User-friendly UI for uploading PDFs and asking questions in real time.
* **Document Processing & Chunking**: Extracts text from PDFs and splits it into manageable chunks using LangChain text splitters.
* **Semantic Vector Search**: Utilizes HuggingFace `sentence-transformers/all-MiniLM-L6-v2` embeddings stored in a local **FAISS** vector database.
* **Context-Aware LLM Answering**: Leverages Google's `flan-t5-base` model to generate concise, accurate answers grounded in document context.
* **Source Transparency**: Displays exact retrieved context chunks for each answer to ensure verification and prevent hallucinations.

---

## 🏗️ System Architecture

```
[ PDF Document Upload ]
          ↓
[ PyPDFLoader & Text Splitter ] (chunk_size=500, overlap=50)
          ↓
[ HuggingFace Embeddings ] (all-MiniLM-L6-v2)
          ↓
[ FAISS Vector Store ]
          ↓
[ Similarity Search ] (Top-5 relevant chunks)
          ↓
[ FLAN-T5 LLM QA Pipeline ]
          ↓
[ Streamlit Interactive Response + Context Expander ]
```

---

## 🛠️ Tech Stack

* **Frontend / UI**: Streamlit
* **RAG Framework**: LangChain (`langchain`, `langchain-community`, `langchain-huggingface`)
* **Vector DB**: FAISS (`faiss-cpu`)
* **Embeddings & LLM**: HuggingFace Transformers (`sentence-transformers/all-MiniLM-L6-v2`, `google/flan-t5-base`)
* **Language**: Python 3.10+

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/Prakher333/rag-document-ai-assistant.git
cd rag-document-ai-assistant
```

### 2. Install Dependencies
It is recommended to use a virtual environment:
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Run the Streamlit App
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 📁 Repository Structure

```
├── app.py              # Streamlit Web Application interface
├── rag.py              # RAG Pipeline (PDF loading, FAISS vector index, FLAN-T5 QA)
├── requirements.txt    # Python dependencies optimized for CPU execution
├── README.md           # Project documentation
├── .gitignore          # Git ignore rules for cache & virtual environments
└── .devcontainer/      # VS Code Dev Container configuration
```

---

## 👤 Author

**Prakher Dwivedi**  
Mining Engineering, NIT Raipur  
*Research Interests:* Artificial Intelligence, Machine Learning, Mining Informatics
