import streamlit as st

from pdf_loader import load_pdf
from chunking import create_chunks
from vector_store import build_vector_store, retrieve_docs
from rag import generate_answer

# Page config
st.set_page_config(
    page_title="Verdict AI",
    page_icon="🧠",
    layout="wide"
)

# Title
st.title("🧠 Verdict AI")
st.caption("Explainable Document Intelligence")

# Upload PDF
uploaded_file = st.file_uploader(
    "Upload PDF",
    type=["pdf"]
)

if uploaded_file:

    # Read PDF
    text = load_pdf(uploaded_file)

    # Create chunks
    chunks = create_chunks(text)

    # Create vector database
    vectorstore = build_vector_store(chunks)

    st.success("PDF loaded successfully!")

    # Ask Question
    question = st.text_input(
        "Ask a question about the PDF"
    )

    if question:

        # Retrieve relevant chunks
        docs = retrieve_docs(
            question,
            vectorstore
        )

        # Generate answer
        answer = generate_answer(
            question,
            docs
        )

        st.subheader("📌 Summary")
        for i, doc in enumerate(docs[:5]):
            point = doc.page_content[:120].replace("\n", " ")
            st.markdown(f"• {point}...")
        st.write(answer)
        with st.expander("📚 View Sources"):
            for i, doc in enumerate(docs):
                st.write(f"Source {i+1}")

        st.info(doc.page_content[:300] + "...")

        st.subheader("📚 Sources")

        for i, doc in enumerate(docs):
            st.write(f"Source {i+1}")
            st.info(doc.page_content[:200] + "...")