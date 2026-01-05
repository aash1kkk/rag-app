import os
import time
import asyncio
from pathlib import Path

import streamlit as st
import inngest
from dotenv import load_dotenv

# -------------------------------------------------
# Load env vars (Railway injects them automatically)
# -------------------------------------------------
load_dotenv()

# -------------------------------------------------
# Streamlit config
# -------------------------------------------------
st.set_page_config(
    page_title="RAG Ingest PDF",
    page_icon="📄",
    layout="centered",
)

# -------------------------------------------------
# Inngest client
# -------------------------------------------------
@st.cache_resource
def get_inngest_client() -> inngest.Inngest:
    event_key = os.getenv("INNGEST_EVENT_KEY")
    if not event_key:
        st.error("❌ INNGEST_EVENT_KEY is not set")
        raise RuntimeError("Missing INNGEST_EVENT_KEY")

    return inngest.Inngest(
        app_id="rag_app",
        event_key=event_key,
        is_production=True,
    )

# -------------------------------------------------
# Async helper (Streamlit-safe)
# -------------------------------------------------
def run_async(coro):
    return asyncio.run(coro)

# -------------------------------------------------
# Helpers
# -------------------------------------------------
def save_uploaded_pdf(file) -> Path:
    uploads_dir = Path("/tmp/uploads")  # Railway-safe temp dir
    uploads_dir.mkdir(parents=True, exist_ok=True)
    path = uploads_dir / file.name
    path.write_bytes(file.getbuffer())
    return path

# -------------------------------------------------
# UI: Upload PDF
# -------------------------------------------------
st.title("📄 Upload a PDF to ingest")

uploaded = st.file_uploader(
    "Choose a PDF",
    type=["pdf"],
    accept_multiple_files=False,
)

if uploaded:
    with st.spinner("Uploading and triggering ingestion…"):
        pdf_path = save_uploaded_pdf(uploaded)
        client = get_inngest_client()

        run_async(
            client.send(
                inngest.Event(
                    name="rag/ingest_pdf",
                    data={
                        "pdf_path": str(pdf_path),
                        "source_id": pdf_path.name,
                    },
                )
            )
        )

        time.sleep(0.3)

    st.success(f"✅ Ingestion triggered for **{pdf_path.name}**")
    st.caption("Ingestion runs asynchronously in the background.")

# -------------------------------------------------
# UI: Ask question
# -------------------------------------------------
st.divider()
st.title("❓ Ask a question about your PDFs")

with st.form("rag_query_form"):
    question = st.text_input("Your question")
    top_k = st.number_input(
        "How many chunks to retrieve",
        min_value=1,
        max_value=300,
        value=5,
        step=1,
    )

    submitted = st.form_submit_button("Ask")

    if submitted and question.strip():
        with st.spinner("Sending your question to Inngest…"):
            client = get_inngest_client()

            run_async(
                client.send(
                    inngest.Event(
                        name="rag/query_pdf_ai",
                        data={
                            "question": question.strip(),
                            "top_k": int(top_k),
                        },
                    )
                )
            )

        st.success("✅ Question submitted!")
        st.info(
            "🧠 Your question is being processed asynchronously.\n\n"
            "Check results once the Inngest function completes."
        )
