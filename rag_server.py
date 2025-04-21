#!/usr/bin/env python3
from __future__ import annotations

import os, re, warnings
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

import chromadb
from chromadb.config import Settings, DEFAULT_TENANT, DEFAULT_DATABASE

# Silence LC deprecation warnings
from langchain_core._api.deprecation import LangChainDeprecationWarning
warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

CHROMA_DIR  = "chroma_db"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TITLE_RE    = re.compile(r"title\s+([0-9]{1,2})", re.I)

# Chroma client
chroma_client = chromadb.PersistentClient(
    path=CHROMA_DIR,
    settings=Settings(anonymized_telemetry=False),
    tenant=DEFAULT_TENANT,
    database=DEFAULT_DATABASE,
)

embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

# LLM backend
OLLAMA_URL = os.getenv("OLLAMA_URL")
if OLLAMA_URL:
    from langchain_ollama import OllamaLLM
    llm = OllamaLLM(model=os.getenv("OLLAMA_MODEL", "gemma3:4b"),
                    base_url=OLLAMA_URL,
                    temperature=0.2,
                    streaming=True)
else:
    from langchain_openai import ChatOpenAI
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Set OLLAMA_URL or OPENAI_API_KEY")
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, streaming=True)

PROMPT = PromptTemplate.from_template(
    """You are a legal assistant. Use ONLY the excerpts below to answer,
citing U.S. Code section numbers. If unsure, say so.

Chat history:
{history}

Excerpts:
{context}

Question: {question}
Answer:""")
parser = StrOutputParser()

CHAT_MEMORY: Dict[str, List[tuple[str, str]]] = {}

def detect_title(text: str) -> Optional[str]:
    m = TITLE_RE.search(text)
    if m:
        n = int(m.group(1))
        if 1 <= n <= 50:
            return f"usc{n:02d}"
    return None

def get_retriever(collection: str, k: int, mmr: bool):
    db = Chroma(client=chroma_client,
                collection_name=collection,
                embedding_function=embeddings)
    search_type = "mmr" if mmr else "similarity"
    kwargs = {"k": k, **({"lambda_mult": 0.5} if mmr else {})}
    return db.as_retriever(search_type=search_type, search_kwargs=kwargs)

app = FastAPI(title="US Code RAG Server", version="2.0.1")

class Query(BaseModel):
    question: str
    top_k: int | None = 6
    conversation_id: str | None = None
    mmr: bool | None = False
    stream: bool | None = True

@app.post("/query")
async def rag(q: Query):
    collection = detect_title(q.question) or "usc01"
    history = ""
    if q.conversation_id and q.conversation_id in CHAT_MEMORY:
        history = "\n".join(f"Q: {qq}\nA: {aa}" for qq, aa in CHAT_MEMORY[q.conversation_id][-10:])

    retriever = get_retriever(collection, q.top_k or 6, q.mmr or False)
    docs_scores = retriever.vectorstore.similarity_search_with_score(q.question, k=q.top_k or 6)

    key_terms = set(re.findall(r"[A-Za-z]{4,}", q.question.lower()))
    docs = [d for d, _ in docs_scores if any(w in d.page_content.lower() for w in key_terms)] or [d for d, _ in docs_scores]

    context = "\n---\n".join(f"[§{d.metadata['section_ref']}] {d.page_content}" for d in docs)
    prompt  = PROMPT.format(history=history, context=context, question=q.question)

    async def streamer():
        chunks = []
        async for tok in llm.astream(prompt):
            chunks.append(tok)
            yield tok
        if q.conversation_id:
            CHAT_MEMORY.setdefault(q.conversation_id, []).append((q.question, "".join(chunks)))

    if q.stream:
        return StreamingResponse(streamer(), media_type="text/plain")
    else:
        answer = parser.invoke(llm.invoke(prompt))
        if q.conversation_id:
            CHAT_MEMORY.setdefault(q.conversation_id, []).append((q.question, answer))
        return {"answer": answer, "context_refs": [d.metadata['section_ref'] for d in docs]}

