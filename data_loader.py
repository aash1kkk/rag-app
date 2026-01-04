from google import genai
from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SentenceSplitter
from dotenv import load_dotenv
import os

# Load env variables
load_dotenv()

# Initialize Gemini client
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Gemini embedding model
EMBED_MODEL = "models/text-embedding-004"
EMBED_DIM = 768  # Gemini embedding dimension

splitter = SentenceSplitter(chunk_size=1000, chunk_overlap=200)


def load_and_chunk_pdf(path: str):
    docs = PDFReader().load_data(file=path)
    texts = [d.text for d in docs if getattr(d, "text", None)]
    chunks = []
    for t in texts:
        chunks.extend(splitter.split_text(t))
    return chunks


def embed_texts(texts: list[str]) -> list[list[float]]:
    vectors = []

    for text in texts:
        response = client.models.embed_content(
            model=EMBED_MODEL,
            contents=text
        )
        vectors.append(response.embeddings[0].values)

    return vectors