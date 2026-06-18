import streamlit as st
import fitz  # PyMuPDF
import os
import json
import google.generativeai as genai
import requests
import xml.etree.ElementTree as ET

# =========================
# GEMINI CONFIG
# =========================
# 1. Check if the key exists in Streamlit secrets first, otherwise check the sidebar
if "GEMINI_API_KEY" in st.secrets:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
else:
    GEMINI_API_KEY = st.sidebar.text_input("Enter Gemini API Key", type="password")

# 2. Configure the model if a key is found
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel("gemini-2.5-flash")
else:
    st.sidebar.warning("Please provide a valid Gemini API Key to run.")

# =========================
# STREAMLIT CONFIG
# =========================
st.set_page_config(page_title="AI Research Synthesis Engine", layout="wide")
st.title("🔬 AI Research Synthesis Engine")
st.caption("Assignment 7 - Compliant Multi-Document Synthesis Engine")

os.makedirs("papers", exist_ok=True)

# Initialize Session States safely
if "claims_store" not in st.session_state:
    st.session_state.claims_store = []
if "search_results" not in st.session_state:
    st.session_state.search_results = []

# =========================
# HELPER FUNCTIONS
# =========================
def extract_pdf_text(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        pages = []
        for i in range(min(8, len(doc))):  # Extended slightly for better content capture
            text = doc[i].get_text()
            pages.append((i + 1, text))
        return pages
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return []

def search_arxiv(query, max_results=8):
    url = f"http://export.arxiv.org/api/query?search_query=all:{query}&start=0&max_results={max_results}"
    try:
        response = requests.get(url, timeout=10)
        root = ET.fromstring(response.content)
        papers = []
        for entry in root.findall("{http://www.w3.org/2005/Atom}entry"):
            title = entry.find("{http://www.w3.org/2005/Atom}title").text
            summary = entry.find("{http://www.w3.org/2005/Atom}summary").text
            score = (query.lower() in title.lower()) + (query.lower() in summary.lower())
            
            pdf_link = None
            for link in entry.findall("{http://www.w3.org/2005/Atom}link"):
                if link.attrib.get("title") == "pdf":
                    pdf_link = link.attrib["href"]
                elif link.attrib.get("type") == "application/pdf":
                    pdf_link = link.attrib["href"]
            
            # Fallback if link attribute title missing
            if not pdf_link:
                id_url = entry.find("{http://www.w3.org/2005/Atom}id").text
                pdf_link = id_url.replace("/abs/", "/pdf/") + ".pdf"

            papers.append({
                "title": title.strip().replace("\n", " "),
                "summary": summary.strip().replace("\n", " "),
                "pdf": pdf_link,
                "score": score
            })
        papers.sort(key=lambda x: x["score"], reverse=True)
        return papers
    except Exception as e:
        st.error(f"Search failed: {e}")
        return []

def extract_claims(page_num, text, paper_name):
    prompt = f"""
    Extract data items matching factual assertions/claims from this research paper text.
    Categorize them strictly as 'Finding', 'Hypothesis', or 'Limitation'.

    Return a STRICT JSON list of objects only. No markdown wrappers (like ```json), no conversation text.

    Expected Schema:
    [
      {{
        "claim": "Clear summary of the claim",
        "type": "Finding" | "Hypothesis" | "Limitation",
        "quote": "Exact verbatim string from the text supporting this",
        "page": {page_num},
        "paper": "{paper_name}"
      }}
    ]

    TEXT:
    {text[:3500]}
    """
    try:
        response = model.generate_content(prompt)
        raw = response.text.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        return json.loads(raw)
    except Exception as e:
        return []

def synthesize(all_claims):
    if not all_claims:
        return "No claims available for synthesis."

    prompt = f"""
    You are an advanced academic research synthesis engine. Your goal is to systematically harmonize multiple claims, highlight consensus, and unearth key gaps or contradictions (watch out closely for conflicting data points or assertions between different papers).

    CRITICAL RULES FOR CITATION TRACEABILITY:
    - Every claim, consensus statement, or contradiction you mention MUST be followed by an inline citation format referencing the originating paper name and page number. format: [Paper Name, p. X].
    - Do not invent any outside literature assertions.

    INPUT CLAIMS DATA:
    {json.dumps(all_claims, indent=2)}

    Generate a structured report with these exact headers:
    ## 1. Executive Summary
    ## 2. Structured Themes & Key Findings (With inline citations)
    ## 3. Consensus Areas
    ## 4. Cross-Source Contradictions & Disagreements (Crucial: explicitly point out conflicting data/findings)
    ## 5. Evidence Strength & Research Gaps
    ## 6. Traceable Bibliography Source Index
    """
    try:
        return model.generate_content(prompt).text
    except Exception as e:
        return f"Synthesis execution failed: {e}"

# =========================
# UI IMPLEMENTATION
# =========================
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("🔎 Paper Discovery & Multi-Doc Ingestion")
    query = st.text_input("Enter research question / query")
    
    if st.button("Search Academic Sources"):
        if query:
            with st.spinner("Querying arXiv database..."):
                st.session_state.search_results = search_arxiv(query)
        else:
            st.error("Please insert a search term.")

    if st.session_state.search_results:
        st.write("### Discovery Results")
        for i, p in enumerate(st.session_state.search_results):
            with st.expander(f"{i+1}. {p['title']}"):
                st.write(p["summary"])
                st.caption(f"URL: {p['pdf']}")
                if st.button("Import Paper into Ingestion Workspace", key=f"dl_{i}"):
                    if p["pdf"]:
                        with st.spinner("Downloading PDF target..."):
                            try:
                                res = requests.get(p["pdf"], timeout=15)
                                clean_title = "".join([c for c in p['title'] if c.isalnum() or c in (' ', '_', '-')]).rstrip()[:30]
                                path = f"papers/{clean_title}.pdf"
                                with open(path, "wb") as f:
                                    f.write(res.content)
                                st.success(f"Successfully cached: {clean_title}.pdf")
                            except Exception as e:
                                st.error(f"Could not automatically retrieve PDF file: {e}")
                    else:
                        st.error("Direct PDF link unavailable for this entry.")

with col2:
    st.subheader("📑 Document Processing & Ingestion Pipeline")
    uploaded_files = st.file_uploader("Upload local research PDFs", type=["pdf"], accept_multiple_files=True)
    
    # Track physical file targets in directory workspace
    all_local_papers = [f for f in os.listdir("papers") if f.endswith(".pdf")]
    
    if uploaded_files:
        for file in uploaded_files:
            path = os.path.join("papers", file.name)
            if not os.path.exists(path):
                with open(path, "wb") as f:
                    f.write(file.read())
        all_local_papers = [f for f in os.listdir("papers") if f.endswith(".pdf")]

    if all_local_papers:
        st.write("#### Active Papers Workspace")
        for p_file in all_local_papers:
            with st.container(border=True):
                st.markdown(f"**📄 {p_file}**")
                
                # Processing Buttons executed per document to maintain session state integrity
                if st.button(f"⚡ Fully Process & Extract Claims: {p_file}", key=f"proc_{p_file}"):
                    with st.spinner("Parsing pages & running LLM claim classification..."):
                        pages = extract_pdf_text(os.path.join("papers", p_file))
                        extracted_count = 0
                        
                        for p_num, p_text in pages:
                            claims = extract_claims(p_num, p_text, p_file)
                            if claims:
                                # Remove pre-existing claims for same document to prevent duplication on rerun
                                st.session_state.claims_store = [c for c in st.session_state.claims_store if c['paper'] != p_file]
                                st.session_state.claims_store.extend(claims)
                                extracted_count += len(claims)
                                
                        st.success(f"Extracted {extracted_count} traceable factual claims successfully.")

# =========================
# SYNTHESIS WORKSPACE
# =========================
st.divider()
st.subheader("📊 Cross-Source Synthesis Workspace")

if st.session_state.claims_store:
    with st.expander("🔍 View Extracted Fact Repository"):
        st.dataframe(st.session_state.claims_store, use_container_width=True)
        if st.button("Clear Claims Database"):
            st.session_state.claims_store = []
            st.rerun()

    if st.button("🚀 Generate Compliant Research Brief", type="primary", key="final_brief"):
        with st.spinner("Synthesizing information matrices and identifying conflicts..."):
            brief_output = synthesize(st.session_state.claims_store)
            st.markdown("---")
            st.markdown(brief_output)
else:
    st.info("Ingest documents and click 'Fully Process & Extract Claims' to populate the fact matrix for synthesis.")