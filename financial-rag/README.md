# 📊 HCLTech Financial RAG

> **An end-to-end Retrieval-Augmented Generation (RAG) system for
> querying HCLTech quarterly financial results using local embeddings,
> ChromaDB, a local FLAN-T5 generator, metadata-aware retrieval, and a
> Streamlit interface.**

------------------------------------------------------------------------

## 🧭 Table of Contents

-   [1. Project Overview](#1-project-overview)
-   [2. Problem Statement](#2-problem-statement)
-   [3. What We Built](#3-what-we-built)
-   [4. Why RAG](#4-why-rag)
-   [5. Architecture](#5-architecture)
-   [6. Complete User Query Flow](#6-complete-user-query-flow)
-   [7. Project Structure](#7-project-structure)
-   [8. Technology Stack](#8-technology-stack)
-   [9. Component-by-Component
    Explanation](#9-component-by-component-explanation)
-   [10. Document Ingestion Pipeline](#10-document-ingestion-pipeline)
-   [11. PDF Extraction](#11-pdf-extraction)
-   [12. Chunking](#12-chunking)
-   [13. Embeddings](#13-embeddings)
-   [14. ChromaDB Vector Store](#14-chromadb-vector-store)
-   [15. Retrieval](#15-retrieval)
-   [16. Metadata Filtering](#16-metadata-filtering)
-   [17. RAG Prompt Construction](#17-rag-prompt-construction)
-   [18. LLM Generation](#18-llm-generation)
-   [19. Source and Page Citations](#19-source-and-page-citations)
-   [20. Streamlit Application](#20-streamlit-application)
-   [21. LangGraph: Where It Fits](#21-langgraph-where-it-fits)
-   [22. Why LangGraph Was Not Required in This
    Version](#22-why-langgraph-was-not-required-in-this-version)
-   [23. API Fallback Strategy](#23-api-fallback-strategy)
-   [24. Installation](#24-installation)
-   [25. Running the Project](#25-running-the-project)
-   [26. Testing](#26-testing)
-   [27. Retrieval Quality Lessons](#27-retrieval-quality-lessons)
-   [28. Known Limitations](#28-known-limitations)
-   [29. Future Improvements](#29-future-improvements)
-   [30. RAG Concepts Cheat Sheet](#30-rag-concepts-cheat-sheet)
-   [31. Interview Explanation](#31-interview-explanation)
-   [32. Final Architecture Summary](#32-final-architecture-summary)

------------------------------------------------------------------------

# 1. Project Overview

This project implements a **Financial RAG chatbot** for HCLTech
quarterly financial reports.

The system uses four quarterly financial PDFs:

``` text
HCLTech_Q1_FY26.pdf
HCLTech_Q2_FY26.pdf
HCLTech_Q3_FY26.pdf
HCLTech_Q4_FY26.pdf
```

The documents are converted into searchable chunks, transformed into
numerical embeddings, stored in ChromaDB, retrieved using semantic
similarity, and finally passed to a local language model to generate an
answer.

The application provides a simple Streamlit interface where a user can
ask questions such as:

``` text
What was HCLTech's revenue from operations in Q1 FY26?
```

The system retrieves the relevant financial evidence and generates an
answer together with the source PDF and page number.

------------------------------------------------------------------------

# 2. Problem Statement

Financial reports are usually large PDF documents containing:

-   Revenue
-   Expenses
-   Profit
-   Tax
-   Segment information
-   Earnings per share
-   Dividends
-   Financial notes
-   Auditor information
-   Quarterly and yearly comparisons

Searching these reports manually is slow.

A conventional keyword search also has limitations.

For example:

``` text
User:
"What was the revenue from operations in Q1 FY26?"
```

A normal keyword search may find every occurrence of:

``` text
revenue
operations
quarter
```

but may not understand the relationship between them.

A RAG system combines:

``` text
Semantic Retrieval
        +
Large Language Model
        =
Grounded Question Answering
```

------------------------------------------------------------------------

# 3. What We Built

The final system consists of:

``` text
                 HCLTech Financial RAG
                         │
        ┌────────────────┴────────────────┐
        │                                 │
   OFFLINE PIPELINE                 ONLINE PIPELINE
        │                                 │
        ▼                                 ▼
   Quarterly PDFs                    User Question
        │                                 │
        ▼                                 ▼
   PDF Extraction                 Quarter Selection
        │                                 │
        ▼                                 ▼
      Chunking                    Query Embedding
        │                                 │
        ▼                                 ▼
     Embeddings                      ChromaDB
        │                                 │
        ▼                                 ▼
     ChromaDB                     Relevant Chunks
                                          │
                                          ▼
                                    RAG Prompt
                                          │
                                          ▼
                                      FLAN-T5
                                          │
                                          ▼
                                  Answer + Sources
                                          │
                                          ▼
                                    Streamlit UI
```

------------------------------------------------------------------------

# 4. Why RAG?

## Traditional LLM

A normal LLM receives:

``` text
Question
   ↓
LLM
   ↓
Answer
```

The model may answer from its learned knowledge.

For private or newly released financial documents, this is not reliable.

------------------------------------------------------------------------

## RAG

RAG means:

> **Retrieval-Augmented Generation**

The flow becomes:

``` text
Question
   ↓
Retrieve relevant evidence
   ↓
Add evidence to prompt
   ↓
LLM
   ↓
Grounded answer
```

The LLM does not need to memorize all HCLTech reports.

Instead:

``` text
ChromaDB = FIND THE INFORMATION

LLM = EXPLAIN THE INFORMATION
```

This separation is one of the most important concepts in this project.

------------------------------------------------------------------------

# 5. Architecture

## High-Level Architecture

``` text
                         ┌─────────────────────┐
                         │   HCLTech PDFs      │
                         │ Q1 Q2 Q3 Q4 FY26    │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │    PyPDF Reader     │
                         │ Extract page text   │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │  Text Splitter      │
                         │ 1200 chars / 200    │
                         │ overlap              │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ MiniLM Embeddings   │
                         │ 384-dimensional     │
                         │ vectors              │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │     ChromaDB        │
                         │ Vector Store        │
                         │ 255 chunks          │
                         └──────────┬──────────┘
                                    │
                                    │
                        USER QUERY │
                                    ▼
                         ┌─────────────────────┐
                         │ Quarter Filter     │
                         │ Q1/Q2/Q3/Q4/All    │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Query Embedding     │
                         │ all-MiniLM-L6-v2    │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Semantic Retrieval  │
                         │ ChromaDB top-k       │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Context Construction│
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ RAG Prompt          │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ FLAN-T5 Base        │
                         │ Local Generation    │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Answer + Source     │
                         │ + Page              │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │     Streamlit       │
                         │       UI            │
                         └─────────────────────┘
```

------------------------------------------------------------------------

# 6. Complete User Query Flow

This is the most important flow to understand.

Suppose the user enters:

``` text
What was HCLTech's revenue from operations in Q1 FY26?
```

## Step 1: Streamlit receives the question

The Streamlit application receives:

``` python
question = "What was HCLTech's revenue from operations in Q1 FY26?"
```

The user also selects:

``` text
Q1 FY26
```

------------------------------------------------------------------------

## Step 2: Quarter is converted into a PDF filter

The application maps:

``` text
Q1 FY26
   ↓
HCLTech_Q1_FY26.pdf
```

This is metadata filtering.

Instead of searching all 255 chunks:

``` text
255 chunks
```

we can restrict retrieval to:

``` text
Q1 document chunks
```

------------------------------------------------------------------------

## Step 3: Query becomes an embedding

The question is passed through:

``` text
all-MiniLM-L6-v2
```

The text becomes a numerical vector:

``` text
Question
   ↓
Embedding Model
   ↓
[0.12, -0.08, 0.31, ...]
   ↓
384-dimensional vector
```

The model does not store the answer.

It converts language into a mathematical representation suitable for
similarity search.

------------------------------------------------------------------------

## Step 4: ChromaDB performs semantic search

ChromaDB compares the query vector with stored document vectors.

Conceptually:

``` text
Query Vector
      │
      ├── similarity → Chunk 1
      ├── similarity → Chunk 2
      ├── similarity → Chunk 3
      └── similarity → Chunk N
```

The closest chunks are returned.

For the Q1 revenue question, one retrieved chunk contains:

``` text
Revenue from operations 30,349
```

------------------------------------------------------------------------

## Step 5: Retrieved chunks become context

The system creates:

``` text
Source: HCLTech_Q1_FY26.pdf
Page: 2

Content:
Revenue from operations 30,349 ...
```

This is the **retrieved context**.

------------------------------------------------------------------------

## Step 6: RAG prompt is constructed

The system combines:

``` text
Context
+
Question
+
Instructions
```

Conceptually:

``` text
Context:
HCLTech_Q1_FY26.pdf
Page 2
Revenue from operations 30,349 ...

Question:
What was HCLTech's revenue from operations in Q1 FY26?

Instruction:
Answer using only the context.
```

------------------------------------------------------------------------

## Step 7: FLAN-T5 generates the answer

The local model:

``` text
google/flan-t5-base
```

receives the prompt.

It generates an answer.

The project observed the model producing:

``` text
30349
```

for the direct numerical question.

The application can format that into:

``` text
HCLTech's revenue from operations for Q1 FY26
was ₹30,349 crore.
```

------------------------------------------------------------------------

## Step 8: Source is displayed

The application also keeps:

``` text
Source: HCLTech_Q1_FY26.pdf
Page: 2
```

Therefore the user can trace the answer back to the source document.

This is called **grounding / provenance**.

------------------------------------------------------------------------

# 7. Project Structure

``` text
financial-rag/
│
├── data/
│   ├── HCLTech_Q1_FY26.pdf
│   ├── HCLTech_Q2_FY26.pdf
│   ├── HCLTech_Q3_FY26.pdf
│   └── HCLTech_Q4_FY26.pdf
│
├── chroma_db/
│   └── Persistent ChromaDB data
│
├── index_documents.py
│   └── PDF ingestion + chunking + embeddings + indexing
│
├── retrieve.py
│   └── Retrieval testing/debugging
│
├── retriever.py
│   └── Reusable retrieval function
│
├── generate.py
│   └── RAG generation testing
│
├── app.py
│   └── Streamlit application
│
└── README.md
    └── Project documentation
```

------------------------------------------------------------------------

# 8. Technology Stack

  -----------------------------------------------------------------------
  Component               Technology              Purpose
  ----------------------- ----------------------- -----------------------
  Language                Python                  Main implementation

  PDF extraction          PyPDF                   Read quarterly PDFs

  Chunking                LangChain Text          Split large text into
                          Splitters               manageable pieces

  Embedding               `all-MiniLM-L6-v2`      Convert text to vectors

  Vector database         ChromaDB                Store/search embeddings

  Generation              `google/flan-t5-base`   Generate answers
                                                  locally

  UI                      Streamlit               Web interface

  Environment             Python virtual          Dependency isolation
                          environment             

  Configuration           `.env` during API       Secure API-key
                          experiments             configuration
  -----------------------------------------------------------------------

------------------------------------------------------------------------

# 9. Component-by-Component Explanation

## Python

Python connects the entire pipeline.

It handles:

-   PDF reading
-   preprocessing
-   embedding generation
-   vector database interaction
-   retrieval
-   prompt creation
-   LLM inference
-   UI logic

------------------------------------------------------------------------

## PyPDF

PyPDF reads PDF documents.

Example:

``` python
reader = PdfReader(pdf_file)

for page_number, page in enumerate(
    reader.pages,
    start=1
):
    text = page.extract_text()
```

The important design decision is that the page number is preserved.

Each extracted page becomes:

``` python
{
    "text": "...",
    "source": "HCLTech_Q1_FY26.pdf",
    "page": 2
}
```

This metadata later enables citations.

------------------------------------------------------------------------

# 10. Document Ingestion Pipeline

The ingestion pipeline is an **offline/indexing operation**.

It does not need to execute every time a user asks a question.

``` text
PDF
 ↓
Extract text
 ↓
Split into chunks
 ↓
Generate embeddings
 ↓
Store in ChromaDB
```

The project processed:

``` text
4 PDFs
↓
95 pages
↓
255 chunks
↓
255 embeddings
```

The embedding vectors have:

``` text
384 dimensions
```

------------------------------------------------------------------------

# 11. PDF Extraction

The PDFs are read from:

``` text
data/
```

The program loops over:

``` python
data_folder.glob("*.pdf")
```

For every PDF:

``` text
PDF
 ↓
Page 1
Page 2
Page 3
...
```

Each non-empty page is stored with its metadata.

Why preserve page metadata?

Because without it, the system could answer:

``` text
Revenue = 30,349 crore
```

but could not tell the user where the number came from.

With metadata:

``` text
Revenue = 30,349 crore

Source:
HCLTech_Q1_FY26.pdf
Page 2
```

------------------------------------------------------------------------

# 12. Chunking

Large documents should not be sent directly to the embedding model.

Instead, the project uses:

``` python
RecursiveCharacterTextSplitter(
    chunk_size=1200,
    chunk_overlap=200
)
```

## Chunk size

``` text
1200 characters
```

approximately limits the amount of text in one chunk.

## Chunk overlap

``` text
200 characters
```

means neighboring chunks share some text.

Conceptually:

``` text
Document:

[A A A A A A A A A A A A]
          [B B B B B B B B B B B B]
                    [C C C C C C C C C C C C]
```

The overlap helps prevent important information from being split exactly
at a boundary.

------------------------------------------------------------------------

# 13. Embeddings

The project uses:

``` text
all-MiniLM-L6-v2
```

from Sentence Transformers.

It converts text into a vector.

Example:

``` text
"What was revenue in Q1?"
              ↓
      Embedding model
              ↓
[0.21, -0.04, 0.72, ...]
```

The vector has:

``` text
384 dimensions
```

The same embedding model must be used for:

``` text
Document chunks
+
User questions
```

Otherwise their vector representations would not be directly comparable.

------------------------------------------------------------------------

# 14. ChromaDB Vector Store

ChromaDB stores:

``` text
ID
Document
Embedding
Metadata
```

A conceptual record looks like:

``` python
{
    "id": "chunk_25",

    "document":
        "Revenue from operations 30,349...",

    "embedding":
        [0.12, -0.08, ...],

    "metadata":
        {
            "source": "HCLTech_Q1_FY26.pdf",
            "page": 2
        }
}
```

The database is persisted at:

``` text
chroma_db/
```

Therefore the vectors do not need to be regenerated every time the
application starts.

------------------------------------------------------------------------

# 15. Retrieval

The reusable retrieval function is:

``` python
retrieve_chunks(
    question,
    source=None,
    top_k=4
)
```

It performs:

``` text
Question
 ↓
Embedding
 ↓
ChromaDB similarity search
 ↓
Top-k chunks
```

The `top_k` value controls how many candidate chunks are returned.

For example:

``` python
top_k=4
```

means:

``` text
Return the 4 most relevant chunks.
```

------------------------------------------------------------------------

# 16. Metadata Filtering

Pure semantic search initially produced an important problem.

For a Q1 question, retrieval could return Q4 chunks because terms such
as:

``` text
revenue
quarter
financial results
```

were semantically similar.

The solution was to use metadata filtering:

``` python
where={
    "source": "HCLTech_Q1_FY26.pdf"
}
```

Now the retrieval flow becomes:

``` text
Question
   ↓
Identify/Select Q1
   ↓
Filter:
HCLTech_Q1_FY26.pdf
   ↓
Semantic search
   ↓
Relevant Q1 chunks
```

This is a key lesson:

> **Semantic similarity and business constraints should often work
> together.**

Vector search alone does not automatically understand every
domain-specific constraint.

------------------------------------------------------------------------

# 17. RAG Prompt Construction

The retrieved documents are inserted into a prompt.

Conceptually:

``` text
CONTEXT
-------
Source: HCLTech_Q1_FY26.pdf
Page: 2

Revenue from operations 30,349 ...

QUESTION
--------
What was HCLTech's revenue from operations in Q1 FY26?

INSTRUCTION
-----------
Answer using only the context.
```

The context is called **augmented context**.

That is where the "A" in RAG comes from:

``` text
R = Retrieval
A = Augmentation
G = Generation
```

------------------------------------------------------------------------

# 18. LLM Generation

The project uses:

``` text
google/flan-t5-base
```

locally.

The model is loaded using:

``` python
AutoTokenizer
AutoModelForSeq2SeqLM
```

The process is:

``` text
Prompt
 ↓
Tokenizer
 ↓
Token IDs
 ↓
FLAN-T5
 ↓
Generated token IDs
 ↓
Decoder
 ↓
Text answer
```

The reason for using a local model in this implementation was to avoid
depending on paid API credits for generation.

------------------------------------------------------------------------

# 19. Source and Page Citations

Each chunk contains:

``` python
{
    "source": "...",
    "page": ...
}
```

The application displays this information after generation.

Example:

``` text
Answer:
HCLTech's revenue from operations for Q1 FY26
was ₹30,349 crore.

Source:
HCLTech_Q1_FY26.pdf — Page 2
```

This provides traceability.

A financial RAG system should not merely say:

``` text
₹30,349 crore
```

It should ideally also tell the user:

``` text
Where did that number come from?
```

------------------------------------------------------------------------

# 20. Streamlit Application

The Streamlit layer provides the user interface.

The application contains:

``` text
Title
Question input
Quarter selector
Ask button
Answer area
Sources area
```

The quarter selector maps:

``` python
{
    "Q1 FY26": "HCLTech_Q1_FY26.pdf",
    "Q2 FY26": "HCLTech_Q2_FY26.pdf",
    "Q3 FY26": "HCLTech_Q3_FY26.pdf",
    "Q4 FY26": "HCLTech_Q4_FY26.pdf"
}
```

This makes the retrieval more controlled.

------------------------------------------------------------------------

# 21. LangGraph: Where It Fits

## Important clarification

**The current implementation does not actually use LangGraph.**

The working project uses direct Python orchestration:

``` text
Streamlit
   ↓
Retriever
   ↓
Prompt construction
   ↓
FLAN-T5
```

Therefore, it would be technically incorrect to claim:

> "This project uses LangGraph."

Instead, LangGraph is a natural **next architectural evolution**.

------------------------------------------------------------------------

# 22. Why LangGraph Was Not Required in This Version

LangGraph becomes useful when the workflow becomes more complex and
stateful.

Our current pipeline is mostly linear:

``` text
Question
 ↓
Retrieve
 ↓
Generate
 ↓
Answer
```

A graph framework is not necessary just to execute three sequential
functions.

However, imagine the system becomes:

``` text
Question
   ↓
Question Classification
   ↓
 ┌───────────────┬────────────────┐
 │               │                │
Financial     Comparison       Unsupported
Question      Question          Question
 │               │                │
 ↓               ↓                ↓
Retrieve      Multi-query      Refusal
 │               │
 ↓               ↓
Rerank         Retrieve
 │               │
 └───────┬───────┘
         ↓
       Generate
         ↓
      Verify
         ↓
      Citation
         ↓
       Answer
```

This is where LangGraph becomes valuable.

------------------------------------------------------------------------

# 23. How the Same Project Could Become a LangGraph System

A future LangGraph version could contain nodes such as:

``` text
START
  │
  ▼
Question Classifier
  │
  ▼
Quarter Resolver
  │
  ▼
Retriever
  │
  ▼
Reranker
  │
  ▼
Answer Generator
  │
  ▼
Citation Validator
  │
  ▼
END
```

Each node performs one responsibility.

For example:

``` python
def retrieve_node(state):
    ...
```

then:

``` python
def generate_node(state):
    ...
```

and:

``` python
def validate_node(state):
    ...
```

The shared state could contain:

``` python
state = {
    "question": "...",
    "quarter": "Q1 FY26",
    "documents": [...],
    "answer": "...",
    "sources": [...]
}
```

LangGraph would be especially useful if the system needs:

-   Conditional routing
-   Retry logic
-   Verification loops
-   Multiple retrieval strategies
-   Tool calling
-   Stateful conversations
-   Human approval
-   Agentic workflows

------------------------------------------------------------------------

# 24. API Fallback Strategy

During development, cloud APIs were tested for embeddings.

OpenAI returned an insufficient-credit/quota error.

Gemini also encountered API quota exhaustion during batch embedding.

The project therefore moved the embedding stage to:

``` text
all-MiniLM-L6-v2
```

and kept generation local with:

``` text
google/flan-t5-base
```

This made the final core pipeline independent of paid API credits.

A production implementation could use a provider abstraction:

``` text
EmbeddingProvider
      │
      ├── Local MiniLM
      ├── OpenAI
      └── Gemini
```

and:

``` text
GenerationProvider
      │
      ├── Local FLAN-T5
      ├── OpenAI
      ├── Gemini
      └── Groq
```

The application could select a fallback provider if the primary provider
becomes unavailable.

------------------------------------------------------------------------

# 25. Installation

Create a virtual environment:

``` powershell
python -m venv .venv
```

Activate it on Windows:

``` powershell
.venv\Scripts\activate
```

Install the core dependencies:

``` powershell
pip install pypdf
pip install langchain-text-splitters
pip install sentence-transformers
pip install chromadb
pip install transformers
pip install torch
pip install streamlit
```

------------------------------------------------------------------------

# 26. Running the Project

## Step 1: Place PDFs

Put the quarterly PDFs inside:

``` text
data/
```

Expected:

``` text
data/
├── HCLTech_Q1_FY26.pdf
├── HCLTech_Q2_FY26.pdf
├── HCLTech_Q3_FY26.pdf
└── HCLTech_Q4_FY26.pdf
```

------------------------------------------------------------------------

## Step 2: Index the documents

Run:

``` powershell
python index_documents.py
```

Expected processing:

``` text
4 PDFs
↓
95 pages
↓
255 chunks
↓
255 embeddings
↓
ChromaDB
```

------------------------------------------------------------------------

## Step 3: Test retrieval

Run:

``` powershell
python retriever.py
```

or:

``` powershell
python retrieve.py
```

This allows retrieval to be tested independently from generation.

------------------------------------------------------------------------

## Step 4: Test generation

Run:

``` powershell
python generate.py
```

This tests:

``` text
Retrieval
+
Prompt
+
LLM
```

without Streamlit.

------------------------------------------------------------------------

## Step 5: Start the UI

Run:

``` powershell
streamlit run app.py
```

Open:

``` text
http://localhost:8501
```

------------------------------------------------------------------------

# 27. Testing

A good RAG system must be tested at multiple levels.

## Test 1: Direct factual question

``` text
What was HCLTech's revenue from operations in Q1 FY26?
```

Expected:

``` text
₹30,349 crore
```

------------------------------------------------------------------------

## Test 2: Another financial metric

``` text
What was HCLTech's profit before tax in Q1 FY26?
```

------------------------------------------------------------------------

## Test 3: Segment question

``` text
What was HCLTech's IT and Business Services revenue in Q1 FY26?
```

------------------------------------------------------------------------

## Test 4: Quarter change

Select:

``` text
Q4 FY26
```

Ask:

``` text
What was HCLTech's revenue from operations in Q4 FY26?
```

This tests metadata filtering.

------------------------------------------------------------------------

## Test 5: Unsupported question

Ask:

``` text
What will HCLTech's revenue be in 2030?
```

The system should not invent a number.

Expected behavior:

``` text
The provided documents do not contain enough
information to answer this question.
```

------------------------------------------------------------------------

## Test 6: Out-of-scope question

Ask:

``` text
What is HCLTech's stock price today?
```

The quarterly PDFs do not provide live market prices.

A grounded RAG system should refuse rather than hallucinate.

------------------------------------------------------------------------

# 28. Retrieval Quality Lessons

One of the most valuable observations during development was that pure
semantic retrieval was not always sufficient.

Initially:

``` text
Q1 question
 ↓
Semantic search over all documents
 ↓
Q4 result
```

This happened because financial reports contain repeated concepts:

``` text
revenue
quarter
financial results
profit
```

The embedding model recognized semantic similarity but did not
automatically enforce:

``` text
Q1 FY26
```

Metadata filtering improved this:

``` text
Q1 question
 ↓
Q1 PDF filter
 ↓
Semantic search
 ↓
Q1 result
```

This demonstrates an important production principle:

> **Use structured constraints together with vector similarity whenever
> the domain provides reliable metadata.**

------------------------------------------------------------------------

# 29. Known Limitations

## 1. PDF text extraction

Financial tables extracted from PDFs can be messy.

For example:

``` text
30,349 30,246 28,057
```

may lose some of the original table structure.

A production system should consider table-aware extraction.

------------------------------------------------------------------------

## 2. Duplicate pages

Some PDFs contain repeated financial-result sections.

Therefore retrieval may return:

``` text
Page 2
Page 17
```

with very similar content.

A production system should add deduplication or parent-document
grouping.

------------------------------------------------------------------------

## 3. Small local LLM

`flan-t5-base` is lightweight and convenient, but it is not a
state-of-the-art financial reasoning model.

It may return:

``` text
30349
```

instead of a polished sentence.

The application can improve this through better prompt engineering or a
stronger generation model.

------------------------------------------------------------------------

## 4. Quarter selection is currently explicit

The current Streamlit UI provides a quarter selector.

A future version can automatically detect:

``` text
Q1
Q2
Q3
Q4
FY26
```

from the user's question.

------------------------------------------------------------------------

## 5. Numerical reasoning

Vector retrieval is good at finding relevant text but is not a
calculator.

For questions involving:

``` text
percentage change
growth rate
difference
CAGR
```

a future system should route numerical calculations to a calculator/tool
node.

------------------------------------------------------------------------

## 6. No full agentic workflow

The current implementation is a straightforward RAG pipeline.

It does not yet contain:

``` text
Planner
Router
Reranker
Verifier
Tool calling
Long-term state
```

Those can be added later using an orchestration framework such as
LangGraph.

------------------------------------------------------------------------

# 30. Future Improvements

## Level 1: Better retrieval

Add:

``` text
Metadata filtering
+
Hybrid search
+
Reranking
+
Deduplication
```

------------------------------------------------------------------------

## Level 2: Better financial extraction

Use table-aware document processing.

Instead of:

``` text
raw PDF text
```

extract structured tables:

``` text
Metric                 Q1 FY26
Revenue                 30,349
Profit before tax        5,189
Tax                      1,345
```

------------------------------------------------------------------------

## Level 3: Automatic quarter detection

Build:

``` text
Question
 ↓
Quarter parser
 ↓
Q1/Q2/Q3/Q4
 ↓
Metadata filter
```

------------------------------------------------------------------------

## Level 4: Numerical reasoning

Add a calculator tool:

``` text
Retrieve financial values
        ↓
Calculator
        ↓
Percentage/difference
        ↓
LLM explanation
```

------------------------------------------------------------------------

## Level 5: Better LLM

Replace FLAN-T5 with a stronger instruction-following model.

Possible architecture:

``` text
Local model
OR
OpenAI
OR
Gemini
OR
Groq
```

behind a common generation interface.

------------------------------------------------------------------------

## Level 6: LangGraph

A production-grade agentic version could be:

``` text
                   START
                     │
                     ▼
              Question Router
                     │
          ┌──────────┼──────────┐
          ▼          ▼          ▼
       Factual   Comparison   Unsupported
          │          │
          ▼          ▼
       Retrieve   Multi-query
          │          │
          └────┬─────┘
               ▼
             Rerank
               │
               ▼
            Generate
               │
               ▼
          Verify Answer
               │
          ┌────┴────┐
          │         │
       Correct   Incorrect
          │         │
          ▼         ▼
       Citation   Retry
          │
          ▼
           END
```

This would turn the project from a simple RAG pipeline into an **agentic
financial research system**.

------------------------------------------------------------------------

# 31. RAG Concepts Cheat Sheet

  Concept             Simple Meaning
  ------------------- --------------------------------------------------
  RAG                 Retrieve information before generating an answer
  Document            Original PDF/report
  Chunk               Small piece of a document
  Embedding           Numerical representation of text
  Vector              Numerical representation used for similarity
  Embedding model     Converts text into vectors
  Vector database     Stores/searches vectors
  ChromaDB            Vector database used here
  Similarity search   Finds semantically related chunks
  Metadata            Information describing a chunk
  Metadata filter     Restricts retrieval using structured information
  Top-k               Number of retrieved candidates
  Context             Evidence given to the LLM
  Prompt              Instructions + context + question
  LLM                 Generates the final response
  Grounding           Keeping the answer tied to retrieved evidence
  Citation            Source/page used for the answer
  Hallucination       Unsupported/generated false information
  Reranking           Reordering retrieved candidates by relevance
  Hybrid search       Combining keyword + vector search
  LangGraph           Framework for stateful/conditional LLM workflows

------------------------------------------------------------------------

# 32. Interview Explanation

If an interviewer asks:

## "Explain your project."

A strong answer is:

> I built a financial RAG system for HCLTech quarterly reports. I
> ingested four quarterly PDFs, extracted the text page by page using
> PyPDF, and split the content into overlapping chunks using LangChain's
> recursive text splitter. I generated 384-dimensional embeddings using
> `all-MiniLM-L6-v2` and stored the chunks, embeddings, and page-level
> metadata in ChromaDB.
>
> When a user asks a question, the question is embedded using the same
> model. The system optionally applies a quarter-level metadata filter,
> performs semantic retrieval in ChromaDB, and retrieves the most
> relevant chunks. Those chunks are then inserted into a grounded prompt
> and passed to a local FLAN-T5 model for generation. The Streamlit
> application displays the answer along with the source PDF and page
> number.
>
> One important issue I encountered was that pure semantic retrieval
> could return a different quarter because financial reports contain
> similar terminology. I solved that by combining semantic retrieval
> with metadata filtering. This improved quarter-specific retrieval
> accuracy.
>
> The current implementation is a linear RAG pipeline. LangGraph was not
> used in the implemented version, but it would be useful if I extended
> the system with question routing, retries, verification, reranking,
> tool calling, or multi-step agentic workflows.

------------------------------------------------------------------------

# 33. Final Architecture Summary

The entire project can be remembered using one diagram:

``` text
                         ┌──────────────┐
                         │ HCLTech PDFs │
                         └──────┬───────┘
                                │
                                ▼
                         ┌──────────────┐
                         │  PDF Reader  │
                         │    PyPDF     │
                         └──────┬───────┘
                                │
                                ▼
                         ┌──────────────┐
                         │   Chunking   │
                         │ 1200 / 200   │
                         └──────┬───────┘
                                │
                                ▼
                         ┌──────────────┐
                         │  Embedding   │
                         │ MiniLM 384D  │
                         └──────┬───────┘
                                │
                                ▼
                         ┌──────────────┐
                         │  ChromaDB    │
                         │ 255 chunks   │
                         └──────┬───────┘
                                │
                                │
                       ┌────────▼────────┐
                       │   USER QUESTION │
                       └────────┬────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │ Query Embedding │
                       └────────┬────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │ Metadata Filter │
                       │ Q1/Q2/Q3/Q4     │
                       └────────┬────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │ Semantic Search │
                       └────────┬────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │ Relevant Chunks │
                       └────────┬────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │  RAG Context    │
                       └────────┬────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │   FLAN-T5       │
                       │   Generation    │
                       └────────┬────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │ Answer + Source │
                       │     + Page      │
                       └────────┬────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │    Streamlit    │
                       │       UI        │
                       └─────────────────┘
```

------------------------------------------------------------------------

# 🎓 The Core Mental Model

If you remember only one thing from this project, remember:

``` text
             RAG
              │
     ┌────────┼────────┐
     │        │        │
     ▼        ▼        ▼
 RETRIEVE  AUGMENT  GENERATE
     │        │        │
     ▼        ▼        ▼
 ChromaDB   Context   FLAN-T5
```

Or even simpler:

``` text
ChromaDB finds the evidence.
        ↓
The prompt gives the evidence to the model.
        ↓
The model explains the evidence.
```

That is the heart of the project.

------------------------------------------------------------------------

## 🚀 Final Project Status

``` text
[████████████████████████████] 100%

PDF ingestion              ✅
Page metadata              ✅
Chunking                   ✅
Embeddings                 ✅
ChromaDB                   ✅
Semantic retrieval         ✅
Metadata filtering         ✅
RAG prompting              ✅
Local LLM generation       ✅
Source/page display        ✅
Streamlit UI               ✅
Retrieval testing          ✅
Hallucination testing      ✅
LangGraph                  🔵 Future extension
Production hardening       🔵 Future extension
```

> **Built as a learning-focused end-to-end RAG system demonstrating
> document ingestion, embeddings, vector search, metadata-aware
> retrieval, grounded generation, and an interactive Streamlit
> interface.**
