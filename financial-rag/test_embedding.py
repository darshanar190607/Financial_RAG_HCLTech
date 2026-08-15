from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()

text = "HCLTech revenue increased during the quarter."

response = client.embeddings.create(
    model="text-embedding-3-small",
    input=text
)

embedding = response.data[0].embedding

print("Embedding created successfully!")
print("Vector length:", len(embedding))
print("First 10 numbers:")
print(embedding[:10])