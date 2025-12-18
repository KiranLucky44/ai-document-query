import streamlit as st
import os
from rag_pipeline import build_rag_pipeline

st.set_page_config(page_title="AI Document Query Assistant")

st.title("📄 AI-Powered Document Query Assistant")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    file_path = f"data/{uploaded_file.name}"

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success("PDF uploaded successfully!")

    qa_chain = build_rag_pipeline(file_path)

    query = st.text_input("Ask a question from the document:")

    if query:
        with st.spinner("Thinking..."):
            response = qa_chain.invoke(query)
            st.write("### Answer:")
            st.write(response)
