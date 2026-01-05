import asyncio
from pathlib import Path
import time
import os

import streamlit as st
import inngest
from dotenv import load_dotenv

load_dotenv()  # Load env vars from Railway or local .env

# ----------------------------
# Streamlit Page Config
# ----------------------------
st.set_page_config(page_title="RAG Ingest PDF", page_icon="📄", layout="centered")

# ----------------------------
# Asyncio loop helper
# ----------------------------
def get_asyncio_loop():
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop

# ----------------------------
# Inngest Client
# ----------------------------
@st.cache_resource
def get_inngest_client() -> inngest.Inngest:
    event_key = os.getenv("INNGEST_EVENT_KEY")
    if not event_key:
        st.error("INNGEST_EVENT_KEY environment variable is missing!")
        raise ValueError("INNGEST_EVENT_KEY is required to send events.")
    return inngest.Inngest(
        app_id="rag_app",
        event_key=event_key,
        is_production=True,
    )

# ----------------------------
# File Upload Handling
# ----------------------------
def save_uploaded_pdf(file) -> Path:
    uploads_dir = Path("/tmp/uploads")
    uploads_dir.mkdir(parents=True, exist_ok=True)
    file_path = uploads_dir / file.name
    file_path.write_bytes(file.getbuffer())
    return file_path

# ----------------------------
# Async Event Trigger
# ----------------------------
async def send_rag_ingest_event(pdf_path: Path) -> None:
    client = get_inngest_client()
    await client.send(
        inngest.Event(
            name="rag/ingest_pdf",
            data={
                "pdf_path": str(pdf_path.resolve()),
                "source_id": pdf_path.name,
            },
        )
    )

async def send_rag_query_event(question: str, top_k: int) -> dict:
    """
    Sends a query event and directly returns the output from the SDK.
    """
    client = get_inngest_client()
    result = await client.send(
        inngest.Event(
            name="rag/query_pdf_ai",
            data={
                "question": question,
                "top_k": top_k,
            },
        )
    )
    # result is already the output dictionary
    return result

# ----------------------------
# PDF Upload UI
# ----------------------------
st.title("Upload a PDF to Ingest")
uploaded = st.file_uploader("Choose a PDF", type=["pdf"], accept_multiple_files=False)

if uploaded is not None:
    with st.spinner("Uploading and triggering ingestion..."):
        path = save_uploaded_pdf(uploaded)
        loop = get_asyncio_loop()
        loop.run_until_complete(send_rag_ingest_event(path))
        time.sleep(0.3)
    st.success(f"Triggered ingestion for: {path.name}")
    st.caption("You can upload another PDF if you like.")

# ----------------------------
# Query Form
# ----------------------------
st.divider()
st.title("Ask a question about your PDFs")

with st.form("rag_query_form"):
    question = st.text_input("Your question")
    top_k = st.number_input("How many chunks to retrieve", min_value=1, max_value=300, value=5, step=1)
    submitted = st.form_submit_button("Ask")

    if submitted and question.strip():
        with st.spinner("Sending event and generating answer..."):
            loop = get_asyncio_loop()
            output = loop.run_until_complete(send_rag_query_event(question.strip(), int(top_k)))
            answer = output.get("answer", "")
            sources = output.get("sources", [])

        st.subheader("Answer")
        st.write(answer or "(No answer)")
        if sources:
            st.caption("Sources")
            for s in sources:
                st.write(f"- {s}")
