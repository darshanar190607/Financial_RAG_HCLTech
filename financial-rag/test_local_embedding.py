from sentence_transformers import SentenceTransformer


model = SentenceTransformer("all-MiniLM-L6-v2")


text = "HCLTech revenue increased during the quarter."

embedding = model.encode(text)


print("Embedding created successfully!")
print("Vector length:", len(embedding))
print("First 10 numbers:")
print(embedding[:10])