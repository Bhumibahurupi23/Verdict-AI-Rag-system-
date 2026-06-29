from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

def build_vector_store(chunks):

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_texts(
        texts=chunks,
        embedding=embedding_model
    )

    return vectorstore

def retrieve_docs(query, vectorstore):

    docs = vectorstore.similarity_search(
        query,
        k=3
    )

    return docs