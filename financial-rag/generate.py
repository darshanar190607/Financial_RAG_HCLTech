from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from retriver import retrieve_chunks


# ==========================================
# 1. LOAD LOCAL LLM
# ==========================================

model_name = "google/flan-t5-base"

tokenizer = AutoTokenizer.from_pretrained(
    model_name
)

model = AutoModelForSeq2SeqLM.from_pretrained(
    model_name
)

print("LLM loaded!")


# ==========================================
# 2. ASK QUESTION
# ==========================================

question = (
    "What was HCLTech's revenue from operations "
    "for Q1 FY26?"
)


# ==========================================
# 3. RETRIEVE RELEVANT CHUNKS
# ==========================================

results = retrieve_chunks(
    question=question,
    source="HCLTech_Q1_FY26.pdf",
    top_k=2
)


# ==========================================
# 4. BUILD CONTEXT
# ==========================================

context_parts = []

for i in range(len(results["documents"][0])):

    text = results["documents"][0][i]

    source = results["metadatas"][0][i]["source"]

    page = results["metadatas"][0][i]["page"]

    context_parts.append(
        f"Source: {source}\n"
        f"Page: {page}\n"
        f"Content:\n{text}"
    )


context = "\n\n".join(context_parts)


# ==========================================
# 5. BUILD RAG PROMPT
# ==========================================

prompt = f"""
Context:
{context}

Question:
{question}

Answer the question using only the context.
Give a short factual answer with the number and unit.
Do not mention the website, company address, or unrelated information.

Answer:
"""


# ==========================================
# 6. TOKENIZE PROMPT
# ==========================================

inputs = tokenizer(
    prompt,
    return_tensors="pt",
    truncation=True,
    max_length=1024
)


# ==========================================
# 7. GENERATE ANSWER
# ==========================================

outputs = model.generate(
    **inputs,
    max_new_tokens=100
)


# ==========================================
# 8. DECODE ANSWER
# ==========================================

answer = tokenizer.decode(
    outputs[0],
    skip_special_tokens=True
)


# ==========================================
# 9. DISPLAY FINAL ANSWER
# ==========================================

print("\n===================================")
print("ANSWER")
print("===================================")

print(
    f"HCLTech's revenue from operations "
    f"for Q1 FY26 was ₹{answer} crore."
)


# ==========================================
# 10. DISPLAY SOURCES
# ==========================================

print("\n===================================")
print("SOURCES")
print("===================================")

seen = set()

for i in range(len(results["documents"][0])):

    source = results["metadatas"][0][i]["source"]

    page = results["metadatas"][0][i]["page"]

    key = (source, page)

    if key not in seen:

        print(
            f"- {source}, Page {page}"
        )

        seen.add(key)