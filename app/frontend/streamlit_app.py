import streamlit as st
import requests

API_URL = "http://localhost:8000/api/query"

st.set_page_config(page_title="LLM Hallucination Detection", layout="wide")
st.title("LLM Hallucination Detection and RAG Validation System")

query = st.text_area("Ask a question", height=120, placeholder="Enter a factual question to test retrieval and hallucination detection.")
if st.button("Submit") and query.strip():
    with st.spinner("Querying the system..."):
        response = requests.post(API_URL, json={"query": query})
        if response.status_code == 200:
            data = response.json()
            st.subheader("Answer")
            st.write(data["answer"])
            st.metric("Confidence Score", f"{data['confidence_score']:.2f}")
            if data["hallucination"]:
                st.warning("Hallucination likely detected. Verify the sources before trusting this answer.")
            else:
                st.success("Answer is grounded and likely reliable.")

            st.subheader("Sources")
            for source in data["sources"]:
                st.write(
                    f"- {source['title']} (score={source['score']:.2f}, source={source.get('source','hybrid')})"
                )
        else:
            st.error(f"API error {response.status_code}: {response.text}")
