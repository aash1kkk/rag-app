import asyncio
from pathlib import Path
import time
import os

import streamlit as st
import inngest
import requests
from dotenv import load_dotenv

load_dotenv()

# ----------------------------
# Streamlit config
# ----------------------------
st.set_page_config(page_title="RAG Ingest PDF", page_icon="📄", layout="centered")

# ----------------------------
# Async loop helper (CRITICAL)
# ----------------------------
def get_asyncio_loop():
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop

# ----------------------------
# Inngest client (EVENT KEY)
# ----------------------------
@st.cache_resource
def get_inngest_client() -> inngest.Inngest:
    event_key = os.getenv("INNGEST_EVENT_KEY")
    if not event_key:
        st.error("INNGEST_EVENT_KEY is missing")
        raise ValueError("Missing INNGEST_EVENT_KEY")

    return inngest.Inngest(
        app_id="rag_app",
        event_key=event_key,
        is_production=True,
    )

# ----------------------------
# File handling
# ----------------------------
def save_uploaded_pdf(file) -> Path:
    uploads = Path("/tmp/uploads")
    uploads.mkdir(parents=True, exist_ok=True)
    path = uploads / file.name
    path.write_bytes(file.getbuffer())
    return path

# ----------------------------
# Send ingest event
# ----------------------------
async def send_rag_ingest_event(pdf_path: Path):
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
# Send query event (RETURNS EVENT IDS)
# ----------------------------
async def send_rag_query_event(question: str, top_k: int):
    client = get_inngest_client()
    return await client.send(
        inngest.Event(
            name="rag/query_pdf_ai",
            data={
                "question": question,
                "top_k": top_k,
            },
        )
    )

# ----------------------------
# Inngest REST API helpers (API TOKEN)
# ----------------------------
INNGEST_API_BASE = "https://api.inngest.com/v1"

def fetch_runs(event_id: str) -> list[dict]:
    api_token = os.getenv("INNGEST_SIGNING_KEY")
    if not api_token:
        raise ValueError("Missing INNGEST_SIGNING_KEY")

    url = f"{INNGEST_API_BASE}/events/{event_id}/runs"
    headers = {"Authorization": f"Bearer {api_token}"}

    resp = requests.get(url, headers=headers)
    resp.raise_for_status()
    return resp.json()["data"]

def wait_for_run_output(event_id: str, timeout=120):
    start = time.time()
    while True:
        runs = fetch_runs(event_id)
        if runs:
            run = runs[0]
            status = run["status"]

            if status in ("Completed", "Succeeded", "Success", "Finished"):
                return run.get("output", {})

            if status in ("Failed", "Cancelled"):
                raise RuntimeError(f"Inngest run {status}")

        if time.time() - start > timeout:
            raise TimeoutError("Timed out waiting for Inngest run")

        time.sleep(1)

# ----------------------------
# UI — Upload PDF
# ----------------------------
st.title("Upload a PDF to Ingest")

uploaded = st.file_uploader("Choose a PDF", type=["pdf"])

if uploaded:
    with st.spinner("Uploading and ingesting..."):
        path = save_uploaded_pdf(uploaded)
        loop = get_asyncio_loop()
        loop.run_until_complete(send_rag_ingest_event(path))
    st.success(f"Ingestion triggered for {path.name}")

# ----------------------------
# UI — Query
# ----------------------------
st.divider()
st.title("Ask a question about your PDFs")

with st.form("query_form"):
    question = st.text_input("Your question")
    top_k = st.number_input("Top K Chunks", 1, 300, 5)
    submitted = st.form_submit_button("Ask")

    if submitted and question.strip():
        with st.spinner("Generating answer..."):
            loop = get_asyncio_loop()

            event_ids = loop.run_until_complete(
                send_rag_query_event(question.strip(), int(top_k))
            )

            event_id = event_ids[0]  # IMPORTANT
            output = wait_for_run_output(event_id)

            answer = output.get("answer", "")
            sources = output.get("sources", [])

        st.subheader("Answer")
        st.write(answer or "(No answer returned)")

        if sources:
            st.caption("Sources")
            for s in sources:
                st.write(f"- {s}")
