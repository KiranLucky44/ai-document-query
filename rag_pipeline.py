import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

def build_rag_pipeline(pdf_path):
    # 1. Load PDF
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # 2. Split text
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    docs = splitter.split_documents(documents)

    # 3. Create embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # 4. Store in FAISS
    vectorstore = FAISS.from_documents(docs, embeddings)

    # 5. Custom Prompt
    prompt_template = """
    You are an AI assistant answering questions from the provided context.
    If the answer is not in the context, say "I don't know".

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    PROMPT = ChatPromptTemplate.from_template(
        template=prompt_template
    )

    # 6. LLM
    llm = HuggingFacePipeline.from_model_id(
        model_id="google/flan-t5-base",
        task="text2text-generation",
        pipeline_kwargs={"temperature": 0, "max_new_tokens": 512}
    )

    # 7. RAG Chain
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    qa_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | PROMPT
        | llm
        | StrOutputParser()
    )

    return qa_chain
