# 📄 AI-Powered Document Query Assistant  
### RAG | LangChain | FAISS | Hugging Face | OpenAI | Streamlit

An AI-driven Document Query Assistant that allows users to upload PDF documents and ask natural language questions. The system uses **Retrieval-Augmented Generation (RAG)** to retrieve relevant document context before generating answers, significantly reducing hallucinations and improving response accuracy.

---

## 🚀 Features
- Upload and query **PDF documents**
- Uses **RAG (Retrieval-Augmented Generation)** for accurate answers
- Vector-based semantic search using **FAISS**
- Embeddings generated via **Hugging Face Sentence Transformers**
- Supports **OpenAI GPT models** or **open-source LLMs**
- Interactive **Streamlit UI**
- Modular and extensible architecture

---

## 🧠 Architecture Overview

1. **Document Loading**  
   PDF documents are loaded and parsed using PyPDF.

2. **Text Chunking**  
   Documents are split into overlapping chunks to preserve context.

3. **Embedding Generation**  
   Each chunk is converted into vector embeddings using Hugging Face models.

4. **Vector Storage**  
   Embeddings are stored and indexed using FAISS for fast similarity search.

5. **Retrieval-Augmented Generation (RAG)**  
   Relevant chunks are retrieved and injected into a custom prompt before sending to the LLM.

6. **Answer Generation**  
   The LLM generates responses grounded in retrieved document context.

---

## 🛠️ Tech Stack

| Component | Technology |
|--------|------------|
| Programming Language | Python 3.9+ |
| LLM Framework | LangChain |
| Vector Database | FAISS |
| Embeddings | Hugging Face (sentence-transformers) |
| LLM | OpenAI GPT / Hugging Face Models |
| UI | Streamlit |
| PDF Processing | PyPDF |

---


