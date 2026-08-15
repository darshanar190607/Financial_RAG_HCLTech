# ==========================================
# IMPORTS
# ==========================================

from pathlib import Path

from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

import chromadb


# ==========================================
# 1. LOAD LOCAL EMBEDDING MODEL
# ==========================================

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

print("Embedding model loaded!")


# ==========================================
# 2. EXTRACT PDF PAGES
# ==========================================

data_folder = Path("data")

all_pages = []

for pdf_file in sorted(data_folder.glob("*.pdf")):

    print(f"Reading: {pdf_file.name}")

    reader = PdfReader(pdf_file)

    for page_number, page in enumerate(reader.pages, start=1):

        text = page.extract_text()

        if text and text.strip():

            all_pages.append({
                "text": text,
                "source": pdf_file.name,
                "page": page_number
            })


print("Total pages:", len(all_pages))


# ==========================================
# 3. CHUNK THE TEXT
# ==========================================

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200,
    chunk_overlap=200
)

chunks = []

for page in all_pages:

    page_chunks = splitter.split_text(page["text"])

    for chunk_text in page_chunks:

        chunks.append({
            "text": chunk_text,
            "source": page["source"],
            "page": page["page"]
        })


print("Total chunks:", len(chunks))


# ==========================================
# 4. CONNECT TO CHROMADB
# ==========================================

chroma_client = chromadb.PersistentClient(
    path="chroma_db"
)

collection = chroma_client.get_or_create_collection(
    name="hcltech_finance"
)

print("ChromaDB connected!")


# ==========================================
# 5. CREATE LOCAL EMBEDDINGS
# ==========================================

def create_embeddings(texts):

    print(
        f"Creating embeddings for {len(texts)} chunks..."
    )

    embeddings = embedding_model.encode(
        texts,
        show_progress_bar=True
    )

    return embeddings.tolist()


# ==========================================
# 6. GET TEXT FROM CHUNKS
# ==========================================

texts = [
    chunk["text"]
    for chunk in chunks
]


# ==========================================
# 7. CREATE EMBEDDINGS
# ==========================================

embeddings = create_embeddings(texts)

print(
    "Total embeddings:",
    len(embeddings)
)

print(
    "Vector dimension:",
    len(embeddings[0])
)


# ==========================================
# 8. PREPARE CHROMADB DATA
# ==========================================

ids = []
documents = []
metadatas = []

for i, chunk in enumerate(chunks):

    ids.append(
        f"chunk_{i}"
    )

    documents.append(
        chunk["text"]
    )

    metadatas.append({
        "source": chunk["source"],
        "page": chunk["page"]
    })


# ==========================================
# 9. STORE EVERYTHING IN CHROMADB
# ==========================================

collection.upsert(
    ids=ids,
    documents=documents,
    embeddings=embeddings,
    metadatas=metadatas
)

print(
    "Data successfully stored in ChromaDB!"
)


# ==========================================
# 10. VERIFY DATABASE
# ==========================================

print(
    "Documents in ChromaDB:",
    collection.count()
)