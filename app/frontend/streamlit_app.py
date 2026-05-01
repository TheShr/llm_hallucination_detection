import os
from urllib.parse import urljoin

import requests
import streamlit as st

API_URL = os.getenv("API_URL") or os.getenv("BACKEND_URL") or "http://localhost:8000/api/query"

st.set_page_config(
    page_title="LLM Hallucination Detection",
    page_icon="🧠",
    layout="wide",
)

st.markdown(
    """
    <style>
        .reportview-container {
            background: #f5f7fb;
        }
        .stApp {
            color: #0f172a;
        }
        .stButton>button {
            background-color: #2563eb;
            color: white;
            border-radius: 0.75rem;
            padding: 0.7rem 1.4rem;
        }
        .stTextInput>div>div>input, .stTextArea>div>div>textarea {
            border-radius: 0.75rem;
            border: 1px solid #cbd5e1;
            padding: 0.85rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("LLM Hallucination Detection")
st.markdown(
    "Use grounded retrieval and automated validation to flag answers that are likely hallucinated."
)

with st.container():
    left, right = st.columns([3, 1])
    with left:
        query = st.text_area(
            "Ask a factual question",
            height=140,
            placeholder="Example: What are the core principles behind the hybrid retrieval approach?",
        )
        submit = st.button("Run verification")

    with right:
        st.subheader("Configuration")
        st.write("Backend endpoint")
        st.code(API_URL)
        st.write("Set `API_URL` or `BACKEND_URL` in your environment for deployments.")

if submit and query.strip():
    api_endpoint = API_URL
    if not api_endpoint.endswith("/query"):
        api_endpoint = urljoin(api_endpoint.rstrip("/"), "/api/query")

    with st.spinner("Analyzing query and verifying grounding..."):
        response = requests.post(api_endpoint, json={"query": query}, timeout=30)

    if response.ok:
        data = response.json()
        columns = st.columns([2, 1])
        with columns[0]:
            st.subheader("Answer")
            st.markdown(f"<div style='font-size:1.05rem; line-height:1.6;'>{data['answer']}</div>", unsafe_allow_html=True)

            st.subheader("Sources")
            for source in data["sources"]:
                st.markdown(
                    f"**{source['title']}**  \n"
                    f"Source: `{source.get('source', 'hybrid')}`  \n"
                    f"Score: `{source['score']:.2f}`"
                )

        with columns[1]:
            st.subheader("Validation")
            st.metric("Confidence score", f"{data['confidence_score']:.2f}")
            if data["hallucination"]:
                st.warning("Hallucination likely detected. Please verify the answer against the sources.")
            else:
                st.success("The answer is grounded and likely reliable.")
            st.progress(min(max(data['confidence_score'], 0.0), 1.0))
    else:
        st.error(f"API error {response.status_code}: {response.text}")
elif submit:
    st.error("Please enter a question before submitting.")
