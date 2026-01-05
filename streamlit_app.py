import asyncio
from pathlib import Path
import time
import os

import streamlit as st
import inngest
from dotenv import load_dotenv
import requests

load_dotenv()  # Load env vars from Railway or local .env

# ----------------------------
# Streamlit Page Config
# ----------------------------
st.set_page_config(page_title="RAG Ingest PDF", page_icon="📄", layout="centered")


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
    uploads_dir = Path("/tmp/uploads")  # Railway ephemeral storage
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


# ----------------------------
# PDF Upload UI
# ----------------------------
st.title("Upload a PDF to Ingest")
uploaded = st.file_uploader("Choose a PDF", type=["pdf"], accept_multiple_files=False)

if uploaded is not None:
    with st.spinner("Uploading and triggering ingestion..."):
        path = save_uploaded_pdf(uploaded)
        loop = asyncio.get_event_loop()
        loop.run_until_complete(send_rag_ingest_event(path))
        time.sleep(0.3)
    st.success(f"Triggered ingestion for: {path.name}")
    st.caption("You can upload another PDF if you like.")

# ----------------------------
# Query Section
# ----------------------------
st.divider()
st.title("Ask a question about your PDFs")


async def send_rag_query_event(question: str, top_k: int) -> str:
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
    return result[0]  # event_id


def _inngest_api_base() -> str:
    return os.getenv("INNGEST_API_BASE", "https://api.inngest.com/v1")


def fetch_runs(event_id: str) -> list[dict]:
    url = f"{_inngest_api_base()}/events/{event_id}/runs"
    event_key = os.getenv("INNGEST_EVENT_KEY")
    if not event_key:
        raise ValueError("INNGEST_EVENT_KEY is required for API requests")

    headers = {"Authorization": f"Bearer {event_key}"}
    resp = requests.get(url, headers=headers)
    resp.raise_for_status()
    return resp.json().get("data", [])


def wait_for_run_output(event_id: str, timeout_s: float = 120.0, poll_interval_s: float = 1.0) -> dict:
    start = time.time()
    last_status = None
    while True:
        runs = fetch_runs(event_id)
        if runs:
            run = runs[0]
            status = run.get("status")
            last_status = status or last_status
            if status in ("Completed", "Succeeded", "Success", "Finished"):
                return run.get("output") or {}
            if status in ("Failed", "Cancelled"):
                raise RuntimeError(f"Function run {status}")
        if time.time() - start > timeout_s:
            raise TimeoutError(f"Timed out waiting for run output (last status: {last_status})")
        time.sleep(poll_interval_s)


# ----------------------------
# Query Form
# ----------------------------
with st.form("rag_query_form"):
    question = st.text_input("Your question")
    top_k = st.number_input("How many chunks to retrieve", min_value=1, max_value=300, value=5, step=1)
    submitted = st.form_submit_button("Ask")

    if submitted and question.strip():
        with st.spinner("Sending event and generating answer..."):
            loop = asyncio.get_event_loop()
            event_id = loop.run_until_complete(send_rag_query_event(question.strip(), int(top_k)))
            output = wait_for_run_output(event_id)
            answer = output.get("answer", "")
            sources = output.get("sources", [])

        st.subheader("Answer")
        st.write(answer or "(No answer)")
        if sources:
            st.caption("Sources")
            for s in sources:
                st.write(f"- {s}")
