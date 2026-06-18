# 🔬 AI Research Synthesis Engine

A multi-document research intelligence system that performs paper discovery, PDF ingestion, claim extraction, cross-document synthesis, and traceable research briefing using LLM-based analysis.

Built for Assignment 7: Research AI Synthesis Engine

---

## 🚀 Features

### 🔎 Paper Discovery (arXiv Integration)
- Query-based academic paper search using arXiv API
- Ranked results based on relevance
- One-click import of research papers into workspace

---

### 📑 Document Ingestion Pipeline
- Upload local PDFs or import from arXiv
- PDF parsing using PyMuPDF
- Multi-page text extraction (optimized for processing efficiency)

---

### 🧠 Claim Extraction Engine
- Extracts factual claims from research papers using LLM
- Classifies each claim into:
  - Finding
  - Hypothesis
  - Limitation
- Stores structured metadata:
  - Supporting quote
  - Page number
  - Source paper

---

### 🔗 Cross-Document Synthesis
- Aggregates claims across multiple papers
- Generates structured comparative research brief:
  - Executive Summary
  - Consensus Areas
  - Contradictions & disagreements
  - Evidence Strength
  - Research Gaps

---

### 📊 Traceable Research Output
All generated insights are traceable to:
- Source paper
- Page number
- Supporting quote

Ensures no unsupported or hallucinated claims are included in final synthesis.

---

## 🏗️ System Architecture

User Query  
→ arXiv Search Engine  
→ Paper Download / Upload  
→ PDF Text Extraction (PyMuPDF)  
→ LLM-based Claim Extraction (Gemini 2.5 Flash)  
→ Structured Claim Store (Session State)  
→ Cross-Document Synthesis Engine  
→ Final Research Brief Generation  

---

## ⚙️ Tech Stack

- Streamlit (Frontend)
- Google Gemini 2.5 Flash (LLM)
- PyMuPDF (PDF parsing)
- arXiv API (paper discovery)
- Python (core orchestration layer)

---

## 🧪 Workflow

1. Enter research query (e.g., *Cassandra vs Bigtable*)
2. System retrieves relevant papers from arXiv
3. Upload or import PDFs into workspace
4. Extract structured text from documents
5. LLM extracts factual claims per page
6. Claims are aggregated across sources
7. Cross-document synthesis generates final research brief

---

## 📌 Output Format

Final research brief includes:
- Executive Summary
- Key Findings
- Consensus Areas
- Contradictions
- Evidence Strength Analysis
- Research Gaps
- Fully traceable citations

---

## 🧠 Core Design Principle

> No claim is valid unless it is traceable to:
- Source paper  
- Page reference  
- Supporting textual evidence  

---

## ⚠️ Limitations

- PDF extraction quality depends on document formatting
- LLM outputs may require JSON sanitization
- arXiv metadata may occasionally be incomplete
- Performance depends on external API latency

---

## 🔮 Future Improvements

- Embedding-based semantic clustering of claims
- Automated contradiction detection system
- Citation graph visualization
- Retrieval-Augmented Generation (RAG) layer
- Multi-model verification for higher reliability

---


---

## 🏁 Status

✔ Paper Discovery  
✔ PDF Ingestion  
✔ Claim Extraction  
✔ Cross-Document Synthesis  
✔ Traceable Citations  
✔ End-to-End Working System  
