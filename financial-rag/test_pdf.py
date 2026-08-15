from pathlib import Path
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ==========================================
# STEP 1: EXTRACT TEXT FROM ALL PDFs
# ==========================================

data_folder = Path("data")

all_pages = []


for pdf_file in sorted(data_folder.glob("*.pdf")):

    print(f"\nReading: {pdf_file.name}")

    reader = PdfReader(pdf_file)

    print(f"Number of pages: {len(reader.pages)}")

    for page_number, page in enumerate(reader.pages, start=1):

        text = page.extract_text()

        if text and text.strip():

            all_pages.append({
                "text": text,
                "source": pdf_file.name,
                "page": page_number
            })


print("\n===================================")
print("TOTAL PAGES EXTRACTED:", len(all_pages))
print("===================================")


# ==========================================
# STEP 2: CREATE CHUNKER
# ==========================================

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200,
    chunk_overlap=200
)


# ==========================================
# STEP 3: CREATE CHUNKS
# ==========================================

chunks = []


for page in all_pages:

    page_chunks = splitter.split_text(page["text"])

    for chunk_text in page_chunks:

        chunks.append({
            "text": chunk_text,
            "source": page["source"],
            "page": page["page"]
        })


# ==========================================
# STEP 4: PRINT RESULTS
# ==========================================

print("\n===================================")
print("TOTAL CHUNKS:", len(chunks))
print("===================================")


for i, chunk in enumerate(chunks[:3]):

    print("\n-----------------------------------")
    print("CHUNK:", i + 1)
    print("SOURCE:", chunk["source"])
    print("PAGE:", chunk["page"])
    print("-----------------------------------")

    print(chunk["text"][:1000])