# ⛏️ Mining Knowledge AI Assistant (RAG + LLM)

An AI-powered document question answering system built using **Retrieval-Augmented Generation (RAG)**, **Jupyter Notebook**, and **Streamlit**.
The system enables users to upload mining engineering documents (PDF format), extract text chunks, generate vector embeddings with persistent disk storage, and interactively ask questions with context-backed AI answers.

---

## 🌟 Key Features

* **Jupyter Notebook Interface (`main.ipynb`)**: Complete notebook workflow for experimentations, document processing, and RAG execution.
* **Interactive Streamlit Web Interface (`app.py`)**: User-friendly UI for uploading PDFs and asking questions in real time.
* **Vector Store Persistence**: Automatically persists local **FAISS** vector indexes to disk for faster re-indexing.
* **Document Processing & Chunking**: Extracts text from PDFs and splits it into manageable chunks using LangChain text splitters.
* **Semantic Vector Search**: Utilizes HuggingFace `sentence-transformers/all-MiniLM-L6-v2` embeddings.
* **Context-Aware LLM Answering**: Leverages Google's `flan-t5-base` model to generate concise, accurate answers grounded in document context.
* **Source Transparency**: Displays exact retrieved context chunks for each answer to ensure verification.

---

## 🏗️ System Architecture

```
[ PDF Document Upload ]
          ↓
[ PyPDFLoader & Text Splitter ] (chunk_size=500, overlap=50)
          ↓
[ HuggingFace Embeddings ] (all-MiniLM-L6-v2)
          ↓
[ FAISS Vector Store ] ──(Persist/Load)──> [ Disk: vector_store/ ]
          ↓
[ Similarity Search ] (Top-5 relevant chunks)
          ↓
[ FLAN-T5 LLM QA Pipeline ]
          ↓
[ Interactive Response / Streamlit UI / Jupyter Notebook ]
```

---

## 🛠️ Tech Stack

* **Notebook**: Jupyter (`main.ipynb`)
* **Frontend / UI**: Streamlit (`app.py`)
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
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Run the App

**Streamlit Web Interface:**
```bash
streamlit run app.py
```

**Jupyter Notebook Interface:**
```bash
jupyter notebook main.ipynb
```

---

## 📁 Repository Structure

```
├── main.ipynb          # Main Jupyter Notebook implementation
├── app.py              # Streamlit Web Application interface
├── rag.py              # RAG Pipeline with persistent FAISS vector storage
├── requirements.txt    # Python dependencies optimized for CPU execution
├── README.md           # Project documentation
└── .gitignore          # Git ignore rules
```

---

## 👤 Author

**Prakher Dwivedi**  
Mining Engineering, NIT Raipur  
*Research Interests:* Artificial Intelligence, Machine Learning, Mining Informatics
