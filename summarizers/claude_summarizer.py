"""
Summarizes research papers to 3 sentences using claude-haiku-4-5.

Strategy:
  1. For arXiv papers: download PDF from arxiv.org/pdf/{arxiv_id}
  2. For conference papers with a URL: attempt direct PDF download
  3. Fallback: summarize from abstract if PDF unavailable
  4. Truncate text to ~60K chars to control cost and fit context window
  5. Store summary in DB; skip papers already summarized
"""

import io
import os
import time
import logging
from typing import Optional

import requests
import anthropic
from pypdf import PdfReader
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

MODEL = "claude-haiku-4-5"
MAX_PDF_CHARS = 60_000   # ~15K tokens — enough to cover most papers
MAX_SUMMARY_TOKENS = 250
DOWNLOAD_TIMEOUT = 30

SUMMARY_PROMPT = """\
You are summarizing a research paper for an expert audience in machine learning and recommender systems.

Read the paper content below and write exactly 3 concise sentences:
1. The core problem or research question addressed.
2. The proposed method or approach.
3. The main experimental results or contributions.

Write only the 3 sentences — no headings, no bullet points, no preamble.

Paper title: {title}

Paper content:
{text}"""


def _pdf_url(paper: dict) -> Optional[str]:
    """Return the best PDF URL for a paper, or None."""
    arxiv_id = paper.get("arxiv_id", "")
    if arxiv_id:
        return f"https://arxiv.org/pdf/{arxiv_id}"

    url = paper.get("url", "")
    if url:
        if url.endswith(".pdf"):
            return url
        # Some S2 / ACM links point to landing pages — skip them
    return None


def _fetch_pdf_text(pdf_url: str) -> Optional[str]:
    """Download a PDF and extract its full text. Returns None on any failure."""
    try:
        resp = requests.get(
            pdf_url,
            timeout=DOWNLOAD_TIMEOUT,
            headers={"User-Agent": "research-paper-crawler/1.0"},
            allow_redirects=True,
        )
        resp.raise_for_status()
        if "pdf" not in resp.headers.get("Content-Type", "").lower():
            return None

        reader = PdfReader(io.BytesIO(resp.content))
        parts = []
        for page in reader.pages:
            parts.append(page.extract_text() or "")
            if sum(len(p) for p in parts) >= MAX_PDF_CHARS:
                break

        text = " ".join(parts)[:MAX_PDF_CHARS].strip()
        return text if text else None

    except Exception as e:
        logger.debug(f"PDF fetch failed ({pdf_url}): {e}")
        return None


def summarize_paper(client: anthropic.Anthropic, paper: dict) -> Optional[str]:
    """Return a 3-sentence summary string, or None if unable."""
    title = paper.get("title", "Untitled")

    # Try full PDF text first
    pdf_url = _pdf_url(paper)
    text = _fetch_pdf_text(pdf_url) if pdf_url else None
    source = "full PDF"

    # Fall back to abstract
    if not text:
        text = (paper.get("abstract") or "").strip()
        source = "abstract"

    if not text:
        logger.info(f"  Skipping (no text): {title[:60]}")
        return None

    logger.debug(f"  Summarizing from {source}: {title[:60]}")
    prompt = SUMMARY_PROMPT.format(title=title, text=text)

    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_SUMMARY_TOKENS,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text.strip()
    except anthropic.RateLimitError:
        logger.warning("  Claude rate limit hit — waiting 60s")
        time.sleep(60)
        return None
    except Exception as e:
        logger.warning(f"  Summarization API error: {e}")
        return None


TOP_N = 10  # Only summarize the top-N highest-scored papers


def run(db, config: dict) -> int:
    """
    Summarize the top-N highest-scored papers that don't have a summary yet.
    Returns the number of summaries written.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY not set. Add it to your .env file:\n"
            "  ANTHROPIC_API_KEY=sk-ant-..."
        )

    client = anthropic.Anthropic(api_key=api_key)

    # papers_needing_summary() returns them sorted by score DESC
    pending = db.papers_needing_summary()[:TOP_N]

    if not pending:
        logger.info(f"Top {TOP_N} papers already summarized.")
        return 0

    logger.info(f"Summarizing top {len(pending)} papers with {MODEL}...")
    written = 0

    for i, paper in enumerate(pending, 1):
        summary = summarize_paper(client, paper)
        if summary:
            db.update_summary(paper["title"], summary)
            written += 1

        if i % 10 == 0:
            logger.info(f"  Progress: {i}/{len(pending)} ({written} summaries written)")

        time.sleep(0.5)  # ~2 req/s to stay well within Haiku rate limits

    logger.info(f"Done. Wrote {written}/{len(pending)} summaries.")
    return written
