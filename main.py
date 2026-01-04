import logging
from fastapi import FastAPI
import inngest
import inngest.fast_api
from inngest.experimental import ai
from dotenv import load_dotenv
import uuid
import os
import datetime
from data_loader import load_and_chunk_pdf , embed_texts
from vecotr_db import QdrantStorage
from customs_types import RAQQueryResult, RAGSearchResult ,RAGUpsertResult,RAGChunkAndSrc
from google import genai
load_dotenv()
inngest_client = inngest.Inngest(
    app_id="rag_app",
    logger=logging.getLogger("uvicorn"),
    is_production=False,
    serializer=inngest.PydanticSerializer()
)

@inngest_client.create_function(
    fn_id="RAG: Ingest PDF",
    trigger = inngest.TriggerEvent(event = "rag/ingest_pdf")

)

async def rag_ingest_pdf(ctx: inngest.Context):
    def _load(ctx: inngest.Context) ->RAGChunkAndSrc:
        pdf_path = ctx.event.data["pdf_path"]
        source_id= ctx.event.data.get("source_id",pdf_path)
        chunks = load_and_chunk_pdf(pdf_path)
        return RAGChunkAndSrc(chunks=chunks,source_id=source_id)

    def _upsert(chunks_and_src: RAGChunkAndSrc) -> RAGUpsertResult:
        chunks = chunks_and_src.chunks
        source_id = chunks_and_src.source_id
        vecs = embed_texts(chunks)
        ids = [str(uuid.uuid5(uuid.NAMESPACE_URL,f"{source_id}:{i}")) for i in range(len(chunks))]
        payloads = [{"source": source_id, "text": chunks[i]} for i in range(len(chunks))]
        QdrantStorage().upsert(ids,vecs,payloads)
        return RAGUpsertResult(ingested=len(chunks))

    chunks_and_src = await ctx.step.run("load-and-chunk",lambda: _load(ctx),output_type =RAGChunkAndSrc)
    ingested = await ctx.step.run("embed-and-upsert",lambda: _upsert(chunks_and_src),output_type =RAGUpsertResult)
    return ingested.model_dump()

@inngest_client.create_function(
     fn_id ="RAG: Query PDF",
     trigger=inngest.TriggerEvent(event="rag/query_pdf_ai")
)
async def rag_query_pdf_ai(ctx: inngest.Context):
    def _search(question: str, top_k: int=10) -> RAGSearchResult:
        query_vec = embed_texts([question])[0]
        store= QdrantStorage()
        found= store.search(query_vec,top_k)
        return RAGSearchResult(contexts=found["contexts"],sources=found["sources"])
    question = ctx.event.data.get("question")
    if not question:
        raise ValueError(f"Missing 'question' in event data: {ctx.event.data}")
    top_k = int(ctx.event.data.get("top_k",10))
    found= await ctx.step.run("embed-and-search",lambda: _search(question,top_k),output_type =RAGSearchResult)

    context_block = "\n\n".join(f"- {c}" for c in found.contexts)


    async def gemini_infer(prompt: str):
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        res = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config={
                "temperature": 0.5,
                "max_output_tokens": 3042,
            }
        )
        return res.text.strip()

    prompt = f"""
    You are a knowledgeable assistant answering questions from a PDF document.

    Use the provided context to give a **detailed, explanatory, and well-structured answer**.
    - Explain ideas step by step
    - Use multiple paragraphs if needed
    - Include all relevant points from the context
    Context:
    {context_block}
    Question:
    {question}
    Answer:
    """

    answer = await ctx.step.run(
        "llm-answer",
        lambda: gemini_infer(prompt)
    )

    return {
        "answer": answer.strip(),
        "sources": found.sources,
        "num_contexts": len(found.contexts)
    }

app = FastAPI()

@app.get("/")
def root():
    return {"status": "ok"}

from inngest.fast_api import serve
serve(app, inngest_client, [rag_ingest_pdf, rag_query_pdf_ai])

