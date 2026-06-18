"""
AI Research Synthesis Engine — Assignment 7
Fixed & Complete Implementation
"""

import streamlit as st
import fitz  # PyMuPDF
import os
import json
import re
import time
import hashlib
import logging
import requests
import xml.etree.ElementTree as ET
from datetime import datetime

# Try new SDK first, fall back to legacy
try:
    from google import genai as genai_new
    from google.genai import types as genai_types
    _USE_NEW_SDK = True
except ImportError:
    import google.generativeai as genai
    _USE_NEW_SDK = False

# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="AI Research Synthesis Engine",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🔬 AI Research Synthesis Engine")
st.caption("Assignment 7 — Multi-Document Traceable Research System")

os.makedirs("papers", exist_ok=True)

# ─────────────────────────────────────────────
# GEMINI CONFIG
# ─────────────────────────────────────────────
if "GEMINI_API_KEY" not in st.secrets:
    st.error("❌ Missing GEMINI_API_KEY in `.streamlit/secrets.toml`")
    st.stop()

GEMINI_MODEL = "gemini-2.5-flash"

if _USE_NEW_SDK:
    _client = genai_new.Client(api_key=st.secrets["GEMINI_API_KEY"])
else:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    _legacy_model = genai.GenerativeModel(GEMINI_MODEL)

# ─────────────────────────────────────────────
# SESSION STATE — safe initialisation
# ─────────────────────────────────────────────
_defaults = {
    "claims_store": [],          # list[dict] — all extracted claims
    "search_results": [],        # list[dict] — arXiv results
    "processed_files": set(),    # set[str]  — already-processed filenames
    "extraction_log": [],        # list[str] — per-page extraction notes
    "synthesis_result": None,    # str       — last synthesis markdown
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ─────────────────────────────────────────────
# HELPERS — JSON extraction (robust)
# ─────────────────────────────────────────────
def extract_json_array(raw: str) -> list:
    """
    Try to pull a JSON array out of a (possibly noisy) LLM response.
    Attempts:
      1. Strip code fences then parse directly.
      2. Regex: first '[' to last ']'.
      3. Line-by-line object extraction fallback.
    Returns a list (possibly empty) — never raises.
    """
    if not raw:
        return []

    # 1. Strip markdown code fences
    cleaned = re.sub(r"```(?:json)?", "", raw).strip()

    # Try direct parse first
    try:
        obj = json.loads(cleaned)
        if isinstance(obj, list):
            return obj
    except json.JSONDecodeError:
        pass

    # 2. Bracket-delimited extraction
    start = cleaned.find("[")
    end = cleaned.rfind("]")
    if start != -1 and end != -1 and end > start:
        try:
            obj = json.loads(cleaned[start : end + 1])
            if isinstance(obj, list):
                return obj
        except json.JSONDecodeError:
            pass

    # 3. Try to extract individual {...} objects and wrap them
    objects = re.findall(r'\{[^{}]+\}', cleaned, re.DOTALL)
    parsed = []
    for obj_str in objects:
        try:
            parsed.append(json.loads(obj_str))
        except json.JSONDecodeError:
            continue
    if parsed:
        return parsed

    return []


def call_llm(prompt: str, retries: int = 3, delay: float = 3.0) -> str:
    """
    Wrapper around Gemini with retry + 429 rate-limit backoff.
    Handles both new google-genai SDK and legacy google-generativeai SDK.
    Returns raw text or raises RuntimeError so callers can surface the error.
    """
    for attempt in range(1, retries + 1):
        try:
            if _USE_NEW_SDK:
                response = _client.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=prompt,
                )
                return response.text or ""
            else:
                response = _legacy_model.generate_content(prompt)
                return response.text or ""

        except Exception as e:
            err_str = str(e)
            logger.warning(f"LLM call failed (attempt {attempt}/{retries}): {err_str[:200]}")

            # Rate-limit: parse retry_delay from error if available
            if "429" in err_str or "quota" in err_str.lower() or "rate" in err_str.lower():
                # Try to extract suggested wait time from error message
                wait_match = re.search(r'retry.*?(\d+)', err_str, re.IGNORECASE)
                wait_secs = int(wait_match.group(1)) if wait_match else 65
                wait_secs = min(wait_secs, 70)  # cap at 70s

                msg = (
                    f"⏳ Rate limit hit (attempt {attempt}/{retries}). "
                    f"Waiting {wait_secs}s before retry…"
                )
                logger.warning(msg)
                st.toast(msg, icon="⏳")

                if attempt < retries:
                    time.sleep(wait_secs)
                else:
                    raise RuntimeError(
                        f"🚫 Gemini API quota exhausted (free tier: 20 req/day). "
                        f"Wait until midnight or upgrade your API plan.\n\nDetails: {err_str[:300]}"
                    )
            else:
                # Transient error — shorter wait
                if attempt < retries:
                    time.sleep(delay * attempt)
                else:
                    raise RuntimeError(f"LLM call failed after {retries} attempts: {err_str[:300]}")

    return ""


# ─────────────────────────────────────────────
# PDF INGESTION
# ─────────────────────────────────────────────
def extract_pdf_pages(pdf_path: str, max_pages: int = 12) -> list[tuple[int, str]]:
    """
    Returns list of (page_number_1indexed, clean_text).
    Skips pages with too little content or garbage indicators.
    """
    pages = []
    try:
        doc = fitz.open(pdf_path)
        for i in range(min(max_pages, len(doc))):
            raw = doc[i].get_text("text")

            # Clean whitespace
            text = re.sub(r'\s+', ' ', raw).strip()

            # Skip nearly-empty pages
            if len(text) < 120:
                logger.info(f"  Page {i+1}: skipped (too short, {len(text)} chars)")
                continue

            # Skip pages that look like pure references/bibliography
            ref_indicators = text.lower()
            if re.match(r'^\s*references?\s*$', ref_indicators[:30]):
                logger.info(f"  Page {i+1}: skipped (references page)")
                continue

            pages.append((i + 1, text))
    except Exception as e:
        logger.error(f"PDF extraction error for {pdf_path}: {e}")

    return pages


# ─────────────────────────────────────────────
# ARXIV SEARCH WITH RELEVANCE RANKING
# ─────────────────────────────────────────────
def score_relevance(title: str, summary: str, query: str) -> float:
    """
    Simple TF-style relevance score: count query token hits
    in title (weighted 2x) and summary.
    """
    tokens = re.findall(r'\w+', query.lower())
    title_l = title.lower()
    summary_l = summary.lower()

    score = 0.0
    for tok in tokens:
        score += 2.0 * title_l.count(tok)
        score += 1.0 * summary_l.count(tok)
    return score


def search_arxiv(query: str, max_results: int = 12, top_k: int = 6) -> list[dict]:
    """
    Query arXiv, rank by relevance to query, return top_k.
    Filters out papers with zero relevance score.
    """
    url = (
        f"https://export.arxiv.org/api/query"
        f"?search_query=all:{requests.utils.quote(query)}"
        f"&start=0&max_results={max_results}"
        f"&sortBy=relevance&sortOrder=descending"
    )
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
    except Exception as e:
        logger.error(f"arXiv request failed: {e}")
        return []

    root = ET.fromstring(resp.content)
    ns = "{http://www.w3.org/2005/Atom}"
    papers = []

    for entry in root.findall(f"{ns}entry"):
        title_el = entry.find(f"{ns}title")
        summary_el = entry.find(f"{ns}summary")
        if title_el is None or summary_el is None:
            continue

        title = re.sub(r'\s+', ' ', title_el.text or "").strip()
        summary = re.sub(r'\s+', ' ', summary_el.text or "").strip()

        pdf_link = None
        for link in entry.findall(f"{ns}link"):
            href = link.attrib.get("href", "")
            if "pdf" in href:
                pdf_link = href
                break
            # fallback: abs link → convert to pdf
            if "/abs/" in href and pdf_link is None:
                pdf_link = href.replace("/abs/", "/pdf/")

        score = score_relevance(title, summary, query)
        if score == 0:
            continue  # noise filter

        papers.append({
            "title": title,
            "summary": summary,
            "pdf": pdf_link,
            "score": score,
        })

    # Sort descending by relevance, return top_k
    papers.sort(key=lambda x: x["score"], reverse=True)
    return papers[:top_k]


# ─────────────────────────────────────────────
# CLAIM EXTRACTION — DETERMINISTIC + RETRY
# ─────────────────────────────────────────────
CLAIM_PROMPT_TEMPLATE = """\
You are a strict academic claim extraction engine.

## TASK
Extract every factual claim from the TEXT below.

## STRICT RULES
1. Output ONLY a valid JSON array — no prose, no markdown, no explanation.
2. Each element MUST match this exact schema:
   {{
     "claim":  "<concise factual statement, max 40 words>",
     "type":   "<Finding | Hypothesis | Limitation>",
     "quote":  "<verbatim sentence(s) from text that support this claim>",
     "page":   {page_num},
     "paper":  "{paper_name}"
   }}
3. Ignore: headers, footers, page numbers, author lists, reference entries,
   acknowledgements, table of contents lines.
4. Do NOT invent or paraphrase beyond the text.
5. If genuinely no claims exist, return exactly: []
6. Include ALL valid claims — do not truncate.

## TEXT (page {page_num})
{text}

## OUTPUT (JSON array only):
"""

CLAIM_PROMPT_STRICT = """\
You are a precision academic extraction engine. Previous extraction returned zero results.
Try again with MAXIMUM sensitivity — extract EVERY sentence that contains a factual
assertion, result, limitation, or hypothesis.

## RULES (same as before)
- Output ONLY a valid JSON array.
- Schema per element:
  {{
    "claim":  "<factual statement>",
    "type":   "<Finding | Hypothesis | Limitation>",
    "quote":  "<verbatim text>",
    "page":   {page_num},
    "paper":  "{paper_name}"
  }}
- Do NOT return empty if there is ANY scientific content.

## TEXT (page {page_num})
{text}

## OUTPUT:
"""


def _validate_claims(raw_list: list, paper_name: str, page_num: int) -> list[dict]:
    """
    Validate and clean a list of raw parsed claim dicts.
    Enforces required fields; fixes page/paper if wrong.
    """
    valid = []
    required = {"claim", "type", "quote", "page", "paper"}
    allowed_types = {"Finding", "Hypothesis", "Limitation"}

    for item in raw_list:
        if not isinstance(item, dict):
            continue
        if not required.issubset(item.keys()):
            continue
        if not item.get("claim", "").strip():
            continue
        if not item.get("quote", "").strip():
            continue

        # Normalise type
        t = item.get("type", "Finding")
        item["type"] = t if t in allowed_types else "Finding"

        # Enforce correct page + paper (LLM sometimes hallucinates these)
        item["page"] = page_num
        item["paper"] = paper_name

        # Deduplicate via claim text hash
        item["_id"] = hashlib.md5(
            f"{paper_name}|{page_num}|{item['claim']}".encode()
        ).hexdigest()[:10]

        valid.append(item)

    return valid


def extract_claims(text: str, paper_name: str, page_num: int) -> list[dict]:
    """
    Extract claims from a single page.
    - First attempt with standard prompt.
    - If zero results, retry once with stricter prompt.
    - On rate-limit error, surfaces the error via st.error and re-raises
      so the ingestion loop stops cleanly instead of showing 0 claims.
    """
    text_chunk = text[:3000]  # stay within context

    # ── Attempt 1 ──
    try:
        prompt1 = CLAIM_PROMPT_TEMPLATE.format(
            page_num=page_num,
            paper_name=paper_name,
            text=text_chunk,
        )
        raw1 = call_llm(prompt1)
        claims1 = _validate_claims(extract_json_array(raw1), paper_name, page_num)
    except RuntimeError as e:
        st.error(str(e))
        raise  # stop the ingestion loop

    if claims1:
        logger.info(f"  [{paper_name}] p{page_num}: {len(claims1)} claims (attempt 1)")
        return claims1

    # ── Attempt 2 — stricter prompt ──
    logger.warning(f"  [{paper_name}] p{page_num}: 0 claims on attempt 1, retrying…")
    try:
        prompt2 = CLAIM_PROMPT_STRICT.format(
            page_num=page_num,
            paper_name=paper_name,
            text=text_chunk,
        )
        time.sleep(1.5)  # brief pause before retry
        raw2 = call_llm(prompt2)
        claims2 = _validate_claims(extract_json_array(raw2), paper_name, page_num)
    except RuntimeError as e:
        st.error(str(e))
        raise

    if claims2:
        logger.info(f"  [{paper_name}] p{page_num}: {len(claims2)} claims (attempt 2)")
    else:
        logger.warning(f"  [{paper_name}] p{page_num}: 0 claims after 2 attempts — logged")
        st.session_state.extraction_log.append(
            f"⚠️ Zero claims after 2 attempts: {paper_name}, page {page_num}"
        )

    return claims2


# ─────────────────────────────────────────────
# DEDUPLICATION
# ─────────────────────────────────────────────
def deduplicate_claims(claims: list[dict]) -> list[dict]:
    """Remove duplicate claims by their _id hash."""
    seen = set()
    unique = []
    for c in claims:
        cid = c.get("_id") or hashlib.md5(
            f"{c.get('paper')}|{c.get('page')}|{c.get('claim')}".encode()
        ).hexdigest()[:10]
        if cid not in seen:
            seen.add(cid)
            unique.append(c)
    return unique


# ─────────────────────────────────────────────
# CROSS-DOCUMENT SYNTHESIS
# ─────────────────────────────────────────────
SYNTHESIS_PROMPT = """\
You are a senior research analyst producing a formal multi-document synthesis report.

## ABSOLUTE RULES
1. EVERY statement you make MUST be followed immediately by its citation in the form:
   [Paper Name, p. X]
2. If you cannot cite a statement, DO NOT make it.
3. No hallucination. No invention. Use ONLY the claims provided.
4. Contradictions between papers MUST be explicitly named.
5. Evidence strength is determined by:
   - strong  = 3+ independent papers agree
   - medium  = exactly 2 papers agree
   - weak    = single paper mention

## CLAIMS (JSON)
{claims_json}

## OUTPUT FORMAT (use these exact headers in Markdown)

### 1. Executive Summary
<2-4 sentence overview of the body of evidence, all statements cited>

### 2. Key Findings by Theme
<Group related claims under thematic sub-headings.
 Each bullet: finding statement [Paper, p. X]>

### 3. Consensus Areas
<Findings that appear in ≥2 papers — show ALL supporting citations side by side>

### 4. Contradictions & Disagreements
<Where papers directly conflict. Format:
 - **Claim**: [Paper A, p. X] states X, but [Paper B, p. Y] states Y.>

### 5. Evidence Strength Index
<Table or bullet list: claim summary | strength | supporting papers>

### 6. Research Gaps
<What is missing, under-studied, or explicitly flagged as future work, with citations>

### 7. Citation Index
<Alphabetical list of every paper cited, with the page numbers referenced>

---
Produce the full report now:
"""


def synthesize(claims: list[dict]) -> str:
    """Run cross-document synthesis over all extracted claims."""
    if not claims:
        return "⚠️ No claims available for synthesis. Please process at least one paper first."

    # Strip internal _id field before sending to LLM
    clean = [{k: v for k, v in c.items() if k != "_id"} for c in claims]

    prompt = SYNTHESIS_PROMPT.format(
        claims_json=json.dumps(clean, indent=2)[:14000]
    )

    try:
        result = call_llm(prompt, retries=3, delay=3.0)
    except RuntimeError as e:
        return f"❌ Synthesis failed: {e}"
    if not result:
        return "❌ Synthesis returned empty response. Please retry."
    return result


# ─────────────────────────────────────────────
# UI — SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.header("📋 Status")
    st.metric("Total Claims", len(st.session_state.claims_store))
    st.metric("Papers Processed", len(st.session_state.processed_files))

    papers_in_store = sorted({c["paper"] for c in st.session_state.claims_store})
    if papers_in_store:
        st.markdown("**Papers in store:**")
        for p in papers_in_store:
            count = sum(1 for c in st.session_state.claims_store if c["paper"] == p)
            st.markdown(f"- `{p}` — {count} claims")

    if st.session_state.extraction_log:
        with st.expander("⚠️ Extraction Warnings", expanded=False):
            for msg in st.session_state.extraction_log:
                st.write(msg)

    st.divider()
    if st.button("🗑️ Clear All Claims", use_container_width=True):
        st.session_state.claims_store = []
        st.session_state.processed_files = set()
        st.session_state.extraction_log = []
        st.session_state.synthesis_result = None
        st.rerun()


# ─────────────────────────────────────────────
# UI — MAIN TABS
# ─────────────────────────────────────────────
tab_discover, tab_ingest, tab_claims, tab_synthesis = st.tabs([
    "🔎 Discover", "📑 Ingest", "📌 Claims", "📊 Synthesis"
])


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 1 — PAPER DISCOVERY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab_discover:
    st.subheader("🔎 Paper Discovery (arXiv)")
    st.caption("Papers are ranked by relevance score and noise-filtered.")

    col_q, col_n = st.columns([3, 1])
    with col_q:
        query = st.text_input("Research query", placeholder="e.g. distributed key-value store consistency")
    with col_n:
        top_k = st.number_input("Max results", min_value=2, max_value=10, value=6)

    if st.button("🔍 Search arXiv", use_container_width=True):
        if not query.strip():
            st.warning("Enter a query first.")
        else:
            with st.spinner("Searching arXiv…"):
                st.session_state.search_results = search_arxiv(query.strip(), max_results=15, top_k=top_k)
            if not st.session_state.search_results:
                st.warning("No relevant results found. Try a different query.")

    for i, paper in enumerate(st.session_state.search_results):
        with st.expander(f"[Score: {paper['score']:.0f}] {paper['title']}"):
            st.write(paper["summary"][:600] + ("…" if len(paper["summary"]) > 600 else ""))

            if paper.get("pdf"):
                if st.button(f"⬇️ Import PDF", key=f"import_{i}"):
                    with st.spinner(f"Downloading {paper['title'][:40]}…"):
                        try:
                            r = requests.get(paper["pdf"], timeout=30)
                            r.raise_for_status()
                            safe_name = re.sub(r'[^\w\-.]', '_', paper["title"][:50]) + ".pdf"
                            path = f"papers/{safe_name}"
                            with open(path, "wb") as f:
                                f.write(r.content)
                            st.success(f"Saved as `{safe_name}`")
                        except Exception as e:
                            st.error(f"Download failed: {e}")
            else:
                st.info("No direct PDF link available.")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 2 — INGESTION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab_ingest:
    st.subheader("📑 Document Ingestion")

    uploaded_files = st.file_uploader(
        "Upload PDF(s)", type=["pdf"], accept_multiple_files=True
    )
    if uploaded_files:
        for uf in uploaded_files:
            save_path = f"papers/{uf.name}"
            with open(save_path, "wb") as out:
                out.write(uf.read())
        st.success(f"Uploaded {len(uploaded_files)} file(s) to /papers")

    st.divider()
    pdf_files = sorted([f for f in os.listdir("papers") if f.endswith(".pdf")])

    if not pdf_files:
        st.info("No PDFs found. Upload files or import from arXiv.")
    else:
        max_pages = st.slider("Max pages per document", 4, 20, 10)

        for fname in pdf_files:
            already = fname in st.session_state.processed_files
            label = f"✅ {fname}" if already else f"📄 {fname}"

            col_f, col_btn = st.columns([4, 1])
            with col_f:
                st.write(label)
            with col_btn:
                btn_label = "Re-Process" if already else "Process"
                if st.button(btn_label, key=f"proc_{fname}"):
                    fpath = f"papers/{fname}"
                    with st.spinner(f"Extracting text from {fname}…"):
                        pages = extract_pdf_pages(fpath, max_pages=max_pages)

                    if not pages:
                        st.error(f"No usable text found in {fname}. File may be scanned/image-only.")
                        continue

                    st.info(f"Found {len(pages)} usable pages. Extracting claims…")

                    # Remove old claims for this file (re-process support)
                    st.session_state.claims_store = [
                        c for c in st.session_state.claims_store if c.get("paper") != fname
                    ]

                    total_new = 0
                    progress = st.progress(0.0)
                    ingestion_failed = False
                    for idx, (page_num, text) in enumerate(pages):
                        try:
                            new_claims = extract_claims(text, fname, page_num)
                        except RuntimeError:
                            # Error already shown via st.error in extract_claims
                            ingestion_failed = True
                            break
                        st.session_state.claims_store.extend(new_claims)
                        total_new += len(new_claims)
                        progress.progress((idx + 1) / len(pages))

                    # Deduplicate globally
                    st.session_state.claims_store = deduplicate_claims(st.session_state.claims_store)
                    st.session_state.processed_files.add(fname)
                    progress.empty()
                    if ingestion_failed:
                        st.warning(f"⚠️ Partial extraction: {total_new} claims saved before error.")
                    else:
                        st.success(f"✅ Extracted {total_new} claims from {fname}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 3 — CLAIMS BROWSER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab_claims:
    st.subheader("📌 Extracted Claims")

    all_claims = st.session_state.claims_store
    if not all_claims:
        st.info("No claims extracted yet. Process documents in the Ingest tab.")
    else:
        # Filter controls
        col_f1, col_f2 = st.columns(2)
        papers_available = sorted({c["paper"] for c in all_claims})
        types_available = sorted({c["type"] for c in all_claims})

        with col_f1:
            filter_paper = st.multiselect("Filter by paper", papers_available, default=papers_available)
        with col_f2:
            filter_type = st.multiselect("Filter by type", types_available, default=types_available)

        filtered = [
            c for c in all_claims
            if c.get("paper") in filter_paper and c.get("type") in filter_type
        ]

        st.caption(f"Showing {len(filtered)} / {len(all_claims)} claims")

        for c in filtered:
            badge = {"Finding": "🟢", "Hypothesis": "🟡", "Limitation": "🔴"}.get(c["type"], "⚪")
            with st.expander(f"{badge} [{c['paper']}, p.{c['page']}] {c['claim'][:80]}"):
                st.markdown(f"**Type:** `{c['type']}`")
                st.markdown(f"**Paper:** {c['paper']}")
                st.markdown(f"**Page:** {c['page']}")
                st.markdown(f"**Claim:** {c['claim']}")
                st.markdown(f"**Quote:** _{c['quote']}_")

        # Export
        if filtered:
            export_json = json.dumps(
                [{k: v for k, v in c.items() if k != "_id"} for c in filtered],
                indent=2
            )
            st.download_button(
                "⬇️ Export Claims (JSON)",
                data=export_json,
                file_name=f"claims_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
            )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 4 — SYNTHESIS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab_synthesis:
    st.subheader("📊 Cross-Document Research Synthesis")

    all_claims = st.session_state.claims_store
    n_papers = len({c["paper"] for c in all_claims})
    n_claims = len(all_claims)

    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric("Claims", n_claims)
    col_m2.metric("Papers", n_papers)
    col_m3.metric("Types", len({c["type"] for c in all_claims}))

    if n_claims == 0:
        st.info("No claims to synthesise. Process at least one document first.")
    else:
        if n_papers < 2:
            st.warning("⚠️ Only 1 paper in store — cross-paper contradictions won't be detected. "
                       "Import a second paper for full synthesis.")

        if st.button("🧠 Generate Research Brief", use_container_width=True, type="primary"):
            with st.spinner("Synthesising across documents… this may take 30–60 seconds."):
                result = synthesize(all_claims)
                st.session_state.synthesis_result = result

        if st.session_state.synthesis_result:
            st.markdown(st.session_state.synthesis_result)
            st.download_button(
                "⬇️ Export Report (Markdown)",
                data=st.session_state.synthesis_result,
                file_name=f"synthesis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown",
            )
