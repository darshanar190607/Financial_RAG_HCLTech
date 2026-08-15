from sentence_transformers import SentenceTransformer
import chromadb


# ==========================================
# LOAD MODEL
# ==========================================

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# ==========================================
# CONNECT TO CHROMADB
# ==========================================

chroma_client = chromadb.PersistentClient(
    path="chroma_db"
)

collection = chroma_client.get_collection(
    name="hcltech_finance"
)


# ==========================================
# RETRIEVAL FUNCTION
# ==========================================

def retrieve_chunks(
    question,
    source=None,
    top_k=4
):

    # Create question embedding
    question_embedding = embedding_model.encode(
        question
    ).tolist()

    # Build query arguments
    query_args = {
        "query_embeddings": [question_embedding],
        "n_results": top_k,
        "include": [
            "documents",
            "metadatas",
            "distances"
        ]
    }

    # Apply source filter if provided
    if source:
        query_args["where"] = {
            "source": source
        }

    results = collection.query(**query_args)

    return results


# ==========================================
# TEST RETRIEVAL
# ==========================================

if __name__ == "__main__":

    question = (
        "What was HCLTech's revenue from operations "
        "for the quarter ended June 30, 2025?"
    )

    results = retrieve_chunks(
        question=question,
        source="HCLTech_Q1_FY26.pdf",
        top_k=4
    )

    print("\n===================================")
    print("RETRIEVAL RESULTS")
    print("===================================")

    for i in range(len(results["documents"][0])):

        print("\n-----------------------------------")
        print(f"RESULT {i + 1}")
        print("-----------------------------------")

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