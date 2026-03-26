import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class RAGPipeline:
    def __init__(self):
        # Initialize the Embeddings Model
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # Manually load the model to avoid pipeline "Unknown task" string errors in transformers
        model_id = "google/flan-t5-base"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
        
        self.vector_db = None

    def load_and_process_pdf(self, file_path_or_bytes):
        """Loads a PDF file, splits it into chunks and creates a vector store."""
        
        # If it's a bytes object (e.g. from Streamlit), save to temp file
        if hasattr(file_path_or_bytes, 'getvalue'):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(file_path_or_bytes.getvalue())
                file_path = tmp_file.name
        else:
            file_path = file_path_or_bytes

        loader = PyPDFLoader(file_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(docs)

        # Build Vector DB
        self.vector_db = FAISS.from_documents(chunks, self.embeddings)
        
        # Cleanup temp file if created
        if hasattr(file_path_or_bytes, 'getvalue'):
            os.remove(file_path)
            
        return self.vector_db

    def retrieve_context(self, query, k=5):
        """Retrieves top k relevant documents for the query."""
        if not self.vector_db:
            raise ValueError("Vector DB is not initialized. Please load a PDF first.")
        
        docs = self.vector_db.similarity_search(query, k=k)
        return docs

    def generate_answer(self, query):
        """Retrieves context and generates an answer using FLAN-T5."""
        if not self.vector_db:
            raise ValueError("Vector DB is not initialized. Please load a PDF first.")
        
        docs = self.retrieve_context(query, k=5)
        context = "\n".join([doc.page_content for doc in docs])
        
        # Construct the prompt
        prompt = f"Use the context below to answer the question.\n\nContext:\n{context}\n\nQuestion:\n{query}\n\nAnswer:\n"
        
        # Tokenize and Generate
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)
        outputs = self.model.generate(**inputs, max_length=200)
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return {
            "answer": answer,
            "context": docs
        }

# Example Usage:
# rag = RAGPipeline()
# vector_db = rag.load_and_process_pdf("sample.pdf")
# result = rag.generate_answer("What is the main topic?")
# print(result["answer"])
