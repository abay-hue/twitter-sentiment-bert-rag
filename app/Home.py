import streamlit as st, requests, os
st.title("Twitter Sentiment (BERT + RAG) - Demo")

st.subheader("Sentiment")
txt = st.text_area("Enter text")
if st.button("Predict") and txt.strip():
    url = os.environ.get("API_URL","http://localhost:8000/predict")
    try:
        r = requests.post(url, json={"text": txt}, timeout=10).json()
        st.write(r)
    except Exception as e:
        st.error(str(e))

st.divider()
st.subheader("RAG (demo)")
st.write("This is a simple FAISS semantic search stub. Build an index via `python src/rag_index.py` then implement answer synthesis.")
