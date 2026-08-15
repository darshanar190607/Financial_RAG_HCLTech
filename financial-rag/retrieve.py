import chromadb

from sentence_transformers import SentenceTransformer


# ==========================================
# 1. LOAD EMBEDDING MODEL
# ==========================================

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# ==========================================
# 2. CONNECT TO CHROMADB
# ==========================================

chroma_client = chromadb.PersistentClient(
    path="chroma_db"
)

collection = chroma_client.get_collection(
    name="hcltech_finance"
)

print(
    "Documents in database:",
    collection.count()
)


# ==========================================
# 3. USER QUESTION
# ==========================================

question = "What was the revenue from operations for the quarter ended June 30, 2025?"


# ==========================================
# 4. CREATE QUESTION EMBEDDING
# ==========================================

question_embedding = embedding_model.encode(
    question
).tolist()


# ==========================================
# 5. SEARCH CHROMADB
# ==========================================

results = collection.query(
    query_embeddings=[question_embedding],
    n_results=8,
    where={
        "source": "HCLTech_Q1_FY26.pdf"
    },
    include=[
        "documents",
        "metadatas",
        "distances"
    ]
)


# ==========================================
# 6. DISPLAY RESULTS
# ==========================================

for i in range(len(results["documents"][0])):

    print("\n===================================")
    print(f"RESULT {i + 1}")
    print("===================================")

    print(
        "Source:",
        results["metadatas"][0][i]["source"]
    )

    print(
        "Page:",
        results["metadatas"][0][i]["page"]
    )

    print(
        "Distance:",
        results["distances"][0][i]
    )

    print("\nText:")
    print(
        results["documents"][0][i]
    )
    
    # ==========================================
# 7. REMOVE DUPLICATE CHUNKS
# ==========================================

seen = set()

print("\nRETRIEVED CONTEXT")
print("=" * 50)

for i in range(len(results["documents"][0])):

    text = results["documents"][0][i]
    source = results["metadatas"][0][i]["source"]
    page = results["metadatas"][0][i]["page"]

    # Use the text itself to detect duplicates
    if text in seen:
        continue

    seen.add(text)

    print("\n-----------------------------------")
    print(f"RESULT {len(seen)}")
    print("-----------------------------------")

    print("Source:", source)
    print("Page:", page)
    print("Distance:", results["distances"][0][i])

    print("\nText:")
    print(text)