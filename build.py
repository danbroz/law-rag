#!/usr/bin/env python3
"""
build.py  –  Load uscXX.xml files, embed each <section> with
             sentence‑transformers, and persist them in ChromaDB.
"""

import os
import re
import uuid
import logging
from html import unescape

import numpy as np
from sentence_transformers import SentenceTransformer
from lxml import etree
import chromadb
from chromadb.config import Settings, DEFAULT_TENANT, DEFAULT_DATABASE

# ──────────────────────────────────────────────────────────────
# General settings
# ──────────────────────────────────────────────────────────────
DATA_DIR      = "data"          # Directory containing uscXX.xml
BATCH_SIZE    = 16
MODEL_NAME    = "sentence-transformers/all-MiniLM-L6-v2"
VECTOR_DIM    = 384             # dimension for above model
PROGRESS_FILE = "build_progress.txt"
CHROMA_DIR    = "chroma_db"     # persisted DuckDB/Parquet store

# ──────────────────────────────────────────────────────────────
# Init model & Chroma client (one per run)
# ──────────────────────────────────────────────────────────────
model = SentenceTransformer(MODEL_NAME)

client = chromadb.PersistentClient(
    path=CHROMA_DIR,
    settings=Settings(anonymized_telemetry=False),
    tenant=DEFAULT_TENANT,
    database=DEFAULT_DATABASE,
)

# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────
def load_completed_titles(path: str) -> set[int]:
    if not os.path.exists(path):
        return set()
    with open(path, "r", encoding="utf‑8") as f:
        return {int(line.strip()) for line in f if line.strip().isdigit()}


def record_completed_title(path: str, title_num: int) -> None:
    with open(path, "a", encoding="utf‑8") as f:
        f.write(f"{title_num}\n")


def parse_usc_xml(xml_path: str) -> list[dict]:
    try:
        with open(xml_path, "r", encoding="utf‑8") as f:
            raw = f.read()
    except Exception as e:
        logging.error("Error reading %s: %s", xml_path, e)
        return []

    raw = re.sub(r'<\?xml[^>]*\?>', '', raw)
    raw = re.sub(r'<!DOCTYPE.*?\]>', '', raw, flags=re.DOTALL)
    raw = re.sub(r'&(?![#a-zA-Z0-9]+;)', '&amp;', raw)
    raw = unescape(raw)

    wrapped = f"<root>{raw}</root>"
    try:
        root = etree.fromstring(wrapped.encode("utf‑8"), parser=etree.XMLParser(recover=True))
    except Exception as e:
        logging.error("Failed to parse %s: %s", xml_path, e)
        return []

    section_elems = root.xpath("//*[local-name()='section']")
    logging.info("Found %d <section> elements in %s", len(section_elems), os.path.basename(xml_path))

    results = []
    for elem in section_elems:
        section_ref = (elem.xpath(".//*[local-name()='num']")[0].text or "Unknown").strip()
        heading = ""
        h_nodes = elem.xpath(".//*[local-name()='heading']")
        if h_nodes and h_nodes[0].text:
            heading = h_nodes[0].text.strip()

        paras = [
            p.text.strip()
            for p in elem.xpath(
                ".//*[local-name()='p'] | .//*[local-name()='paragraph'] | .//*[local-name()='clause']"
            )
            if p.text and p.text.strip()
        ]
        combined_text = f"{heading}\n" + "\n".join(paras)
        if combined_text.strip():
            results.append({"ref": section_ref, "text": combined_text})

    return results


def insert_batch(collection, sections: list[dict], embeddings: np.ndarray) -> None:
    ids = [str(uuid.uuid4()) for _ in sections]
    documents = [sec["text"] for sec in sections]
    metadatas = [{"section_ref": sec["ref"]} for sec in sections]
    vectors = embeddings.tolist()

    collection.add(ids=ids, documents=documents, metadatas=metadatas, embeddings=vectors)


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────
def main() -> None:
    done_titles = load_completed_titles(PROGRESS_FILE)

    for i in range(1, 51):
        if i in done_titles:
            logging.info("Title %d already processed – skipping.", i)
            continue

        file_suffix = str(i).zfill(2)
        xml_filename = f"usc{file_suffix}.xml"
        xml_path = os.path.join(DATA_DIR, xml_filename)

        if not os.path.exists(xml_path):
            logging.warning("%s not found; marking Title %d as complete.", xml_filename, i)
            record_completed_title(PROGRESS_FILE, i)
            continue

        coll_name = f"usc{file_suffix}"
        collection = client.get_or_create_collection(name=coll_name)
        logging.info("Using Chroma collection '%s'.", coll_name)

        sections = parse_usc_xml(xml_path)
        if not sections:
            logging.info("No sections found in %s; skipping.", xml_filename)
            record_completed_title(PROGRESS_FILE, i)
            continue
        logging.info("Collected %d sections from %s.", len(sections), xml_filename)

        batch_buffer = []
        for sec in sections:
            batch_buffer.append(sec)
            if len(batch_buffer) >= BATCH_SIZE:
                texts = [b["text"] for b in batch_buffer]
                embeddings = model.encode(texts, batch_size=BATCH_SIZE, show_progress_bar=False).astype(np.float32)
                insert_batch(collection, batch_buffer, embeddings)
                batch_buffer.clear()

        if batch_buffer:
            texts = [b["text"] for b in batch_buffer]
            embeddings = model.encode(texts, batch_size=BATCH_SIZE, show_progress_bar=False).astype(np.float32)
            insert_batch(collection, batch_buffer, embeddings)

        logging.info("Finished Title %d into Chroma collection '%s'.", i, coll_name)
        record_completed_title(PROGRESS_FILE, i)

    logging.info("✓ All Titles processed.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    main()

