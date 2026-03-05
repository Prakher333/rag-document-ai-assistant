# Mining Knowledge AI Assistant (RAG + LLM)

An AI-powered document question answering system built using **Retrieval Augmented Generation (RAG)**.
The system retrieves relevant information from mining engineering documents and generates answers using a language model.

## Project Overview

This project implements a **Retrieval Augmented Generation (RAG) pipeline** that allows users to ask questions about mining documents. The system searches relevant sections of the document using vector similarity and then generates answers based on the retrieved context.

The assistant acts as an **AI knowledge retrieval system for technical documents**.

## Features

* Document-based question answering
* Semantic search using vector embeddings
* Context-aware answer generation
* Interactive terminal-based AI assistant
* Retrieval Augmented Generation pipeline

## System Architecture

PDF Document
↓
Text Chunking
↓
Sentence Transformer Embeddings
↓
FAISS Vector Database
↓
Similarity Search
↓
Context Retrieval
↓
QA Model
↓
Generated Answer

## Tech Stack

* Python
* LangChain
* SentenceTransformers
* FAISS Vector Database
* HuggingFace Transformers
* Google Colab

## How It Works

1. A PDF document is loaded and split into smaller chunks.
2. Each chunk is converted into vector embeddings.
3. The embeddings are stored in a **FAISS vector database**.
4. When a user asks a question:

   * The system retrieves the most relevant document chunks.
   * These chunks are passed as context to the language model.
5. The model extracts and returns the most relevant answer.

## Usage

Run the notebook or script and ask questions directly from the terminal.

Example:

Ask your question: What is longwall mining?

Answer: Longwall mining is an underground mining technique where a long wall of coal is mined in a single slice.

Type **exit** to stop the assistant.

## Example Use Cases

* Mining engineering document search
* Technical document QA systems
* Knowledge assistants for domain-specific documents
* AI-powered research assistants

## Future Improvements

* Add conversational memory
* Implement reranking for better retrieval
* Build a web interface using Streamlit
* Add support for multiple documents
* Integrate advanced LLMs for improved reasoning

## Author

Prakher Dwivedi
Mining Engineering, NIT Raipur

Research Interests:
Artificial Intelligence, Machine Learning, Mining Informatics
