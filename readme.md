# 🔬 AI Research Synthesis Engine

A multi-document research intelligence system that performs paper discovery, PDF ingestion, claim extraction, cross-document synthesis, and traceable research briefing using LLM-based analysis.

Built for Assignment 7: Research AI Synthesis Engine

---

## 🚀 Features

### 🔎 Paper Discovery (arXiv Integration)
- Query-based academic paper search using arXiv API
- Ranked results based on relevance
- One-click import of papers into workspace

---

### 📑 Document Ingestion Pipeline
- Upload local PDFs or import from arXiv
- PDF parsing using PyMuPDF
- Multi-page text extraction

---

### 🧠 Claim Extraction Engine
- Extracts factual claims from research papers
- Classifies into:
  - Finding
  - Hypothesis
  - Limitation
- Stores:
  - Supporting quote
  - Page number
  - Source paper

---

### 🔗 Cross-Document Synthesis
- Merges claims across multiple papers
- Generates structured research brief:
  - Executive Summary
  - Consensus
  - Contradictions
  - Evidence Strength
  - Research Gaps

---

### 📊 Traceable Research Output
Every insight includes:
- Paper name
- Page number
- Supporting quote

No unsupported or hallucinated claims are used in synthesis.

---

## 🏗️ System Architecture

User Query  
→ arXiv Search  
→ Paper Download / Upload  
→ PDF Text Extraction (PyMuPDF)  
→ LLM Claim Extraction (Gemini 2.5 Flash)  
→ Claim Store (Session State)  
→ Cross-Document Synthesis  
→ Final Research Brief  

---

## ⚙️ Tech Stack

- Streamlit (Frontend)
- Google Gemini 2.5 Flash (LLM)
- PyMuPDF (PDF parsing)
- arXiv API (paper discovery)
- Python (core logic)

---

## 🧪 Workflow

1. Enter research query (e.g., Cassandra vs Bigtable)
2. System fetches relevant papers from arXiv
3. Upload or import PDFs
4. Extract text from documents
5. LLM extracts structured claims
6. Claims are aggregated
7. Final synthesis generates research brief

---

## 📌 Output Format

Final output includes:
- Executive Summary
- Key Findings
- Consensus Areas
- Contradictions
- Evidence Strength
- Research Gaps
- Traceable citations

---

## 🧠 Core Principle

No claim is generated without:
- Source paper
- Page reference
- Supporting quote

---

## ⚠️ Limitations

- PDF extraction depends on formatting quality
- LLM outputs may require JSON cleanup
- arXiv metadata sometimes incomplete
- API latency affects performance

---

## 🔮 Future Improvements

- Embedding-based claim clustering
- Contradiction detection system
- Citation graph visualization
- RAG-based retrieval layer
- Multi-model verification

---

## 📂 Project Structure
