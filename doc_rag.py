"""A minimal document question-answering (RAG) agent.

Run with: streamlit run app.py
"""
import io
import os
from dataclasses import dataclass

import streamlit as st
from docx import Document
from openai import OpenAI
from pypdf import PdfReader


EMBEDDING_MODEL = "text-embedding-3-small"
ANSWER_MODEL = "gpt-4.1-mini"
CHUNK_SIZE = 1_000       # characters in each searchable unit
CHUNK_OVERLAP = 150      # preserves meaning at chunk boundaries
TOP_K = 5                # number of chunks sent to the answer model


@dataclass
class Chunk:
    text: str
    source: str
    page: int | None
    number: int
    embedding: list[float] | None = None


def read_pdf(file_bytes: bytes, filename: str) -> list[Chunk]:
    """Extract one text item per PDF page, then split it into chunks."""
    reader = PdfReader(io.BytesIO(file_bytes))
    chunks = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        chunks.extend(make_chunks(text, filename, page_number, len(chunks)))
    return chunks


def read_docx(file_bytes: bytes, filename: str) -> list[Chunk]:
    document = Document(io.BytesIO(file_bytes))
    text = "\n".join(paragraph.text for paragraph in document.paragraphs)
    return make_chunks(text, filename, None, 0)


def read_txt(file_bytes: bytes, filename: str) -> list[Chunk]:
    text = file_bytes.decode("utf-8", errors="replace")
    return make_chunks(text, filename, None, 0)


def make_chunks(text: str, source: str, page: int | None, start_number: int) -> list[Chunk]:
    """Split text with overlap. Empty extracted pages are ignored."""
    text = " ".join(text.split())
    if not text:
        return []
    chunks = []
    step = CHUNK_SIZE - CHUNK_OVERLAP
    for index, start in enumerate(range(0, len(text), step)):
        piece = text[start : start + CHUNK_SIZE]
        if piece:
            chunks.append(Chunk(piece, source, page, start_number + index + 1))
    return chunks


def extract_chunks(uploaded_files) -> list[Chunk]:
    all_chunks = []
    for uploaded_file in uploaded_files:
        data = uploaded_file.getvalue()
        suffix = uploaded_file.name.rsplit(".", 1)[-1].lower()
        if suffix == "pdf":
            all_chunks.extend(read_pdf(data, uploaded_file.name))
        elif suffix == "docx":
            all_chunks.extend(read_docx(data, uploaded_file.name))
        elif suffix == "txt":
            all_chunks.extend(read_txt(data, uploaded_file.name))
    return all_chunks


def embed_chunks(client: OpenAI, chunks: list[Chunk]) -> None:
    """Embed in batches to avoid request-size limits."""
    batch_size = 100
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=[chunk.text for chunk in batch],
        )
        for chunk, item in zip(batch, response.data):
            chunk.embedding = item.embedding


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot_product = sum(x * y for x, y in zip(a, b))
    magnitude_a = sum(x * x for x in a) ** 0.5
    magnitude_b = sum(y * y for y in b) ** 0.5
    return dot_product / (magnitude_a * magnitude_b) if magnitude_a and magnitude_b else 0


def retrieve(client: OpenAI, question: str, chunks: list[Chunk]) -> list[Chunk]:
    question_embedding = client.embeddings.create(
        model=EMBEDDING_MODEL, input=question
    ).data[0].embedding
    ranked = sorted(
        chunks,
        key=lambda chunk: cosine_similarity(question_embedding, chunk.embedding or []),
        reverse=True,
    )
    return ranked[:TOP_K]


def answer_question(client: OpenAI, question: str, evidence: list[Chunk]) -> str:
    context = "\n\n".join(
        f"[Source {i}: {chunk.source}" +
        (f", page {chunk.page}" if chunk.page else "") +
        f"]\n{chunk.text}"
        for i, chunk in enumerate(evidence, start=1)
    )
    instructions = """You answer questions using only the supplied document excerpts.
If the excerpts do not contain the answer, say exactly what is missing. Do not use
outside knowledge. Cite factual statements with [Source N]. Be concise and clear."""
    response = client.responses.create(
        model=ANSWER_MODEL,
        instructions=instructions,
        input=f"Question: {question}\n\nDocument excerpts:\n{context}",
    )
    return response.output_text


st.set_page_config(page_title="Document RAG Agent", page_icon="📄")
st.title("📄 Document RAG Agent")
st.caption("Upload documents, then ask questions grounded in their contents.")

with st.sidebar:
    st.header("1. Configure")
    api_key = st.text_input("OpenAI API key", type="password", value=os.getenv("OPENAI_API_KEY", ""))
    files = st.file_uploader("Upload PDFs, DOCX, or TXT", type=["pdf", "docx", "txt"], accept_multiple_files=True)
    index_button = st.button("2. Build document index", type="primary", disabled=not files or not api_key)

if index_button:
    try:
        with st.spinner("Reading and embedding documents..."):
            client = OpenAI(api_key=api_key)
            chunks = extract_chunks(files)
            if not chunks:
                st.error("No readable text was found. This may be a scanned PDF; OCR is needed for image-only PDFs.")
            else:
                embed_chunks(client, chunks)
                st.session_state.chunks = chunks
                st.session_state.api_key = api_key
                st.success(f"Index ready: {len(chunks)} searchable chunks from {len(files)} file(s).")
    except Exception as error:
        st.error(f"Could not build the index: {error}")

question = st.chat_input("3. Ask a question about your documents")
if question:
    if "chunks" not in st.session_state:
        st.warning("Upload files and build the document index first.")
    else:
        try:
            client = OpenAI(api_key=st.session_state.api_key)
            with st.spinner("Retrieving evidence and drafting an answer..."):
                evidence = retrieve(client, question, st.session_state.chunks)
                answer = answer_question(client, question, evidence)
            with st.chat_message("assistant"):
                st.write(answer)
                with st.expander("Retrieved evidence"):
                    for i, chunk in enumerate(evidence, start=1):
                        location = f"page {chunk.page}" if chunk.page else "document text"
                        st.markdown(f"**Source {i}: {chunk.source} ({location})**")
                        st.write(chunk.text)
        except Exception as error:
            st.error(f"Could not answer the question: {error}")
