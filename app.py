
import streamlit as st

st.set_page_config(page_title="Verdict AI Test", layout="wide")

st.title("⚖️ Verdict AI")
st.success("Streamlit is working!")
st.write("We are building your document intelligence dashboard.")

name = st.text_input("Enter a document name")

if name:
    st.write("Document name:", name)
