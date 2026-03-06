"""
Keyword-based relevance scoring and filtering.

Score breakdown:
  Keyword relevance (base):
    - primary keyword in title        -> +5 per match
    - primary keyword in abstract     -> +3 per match
    - secondary keyword in title      -> +2 per match
    - secondary keyword in abstract   -> +1 per match

  CTR/CVR focus bonus:
    - primary keyword appears 3+ times across title+abstract -> +5
    - primary keyword appears 5+ times across title+abstract -> +5 more (cumulative)

  Venue bonus (top conference):
    - RecSys, KDD, WWW, SIGIR         -> +10

  Recency bonus (lower priority):
    - published in current year       -> +3
    - published in previous year      -> +2
    - published 2 years ago           -> +1
"""

import re
import logging
from datetime import datetime
from typing import List, Dict

logger = logging.getLogger(__name__)

TOP_CONFERENCES = {"recsys", "kdd", "www", "sigir"}
CTR_CVR_FOCUS_KEYWORDS = {
    "ctr prediction", "cvr prediction", "click-through rate",
    "conversion rate", "click prediction", "purchase prediction",
}


def _normalize(text: str) -> str:
    return text.lower()


def _count_occurrences(text: str, keyword: str) -> int:
    return text.count(keyword)


def _venue_bonus(paper: Dict) -> int:
    venue = _normalize(paper.get("venue", ""))
    return 10 if any(conf in venue for conf in TOP_CONFERENCES) else 0


def _focus_bonus(title: str, abstract: str) -> int:
    """Bonus for papers that heavily discuss CTR/CVR — multiple mentions signal focus."""
    combined = title + " " + abstract
    total_hits = sum(_count_occurrences(combined, kw) for kw in CTR_CVR_FOCUS_KEYWORDS)
    if total_hits >= 5:
        return 10
    if total_hits >= 3:
        return 5
    return 0


def _recency_bonus(paper: Dict) -> int:
    current_year = datetime.today().year
    year = paper.get("year")
    if not year:
        return 0
    gap = current_year - year
    if gap == 0:
        return 3
    if gap == 1:
        return 2
    if gap == 2:
        return 1
    return 0


def score_paper(paper: Dict, config: dict) -> int:
    primary = [kw.lower() for kw in config["keywords"]["primary"]]
    secondary = [kw.lower() for kw in config["keywords"]["secondary"]]

    title = _normalize(paper.get("title", ""))
    abstract = _normalize(paper.get("abstract", ""))

    base_score = 0

    # Keyword relevance
    for kw in primary:
        if kw in title:
            base_score += 5
        if kw in abstract:
            base_score += 3

    for kw in secondary:
        if kw in title:
            base_score += 2
        if kw in abstract:
            base_score += 1

    # Bonuses only apply if paper has at least some keyword relevance
    if base_score == 0:
        return 0

    bonus = 0
    bonus += _focus_bonus(title, abstract)
    bonus += _venue_bonus(paper)
    bonus += _recency_bonus(paper)  # lowest priority

    return base_score + bonus


def filter_and_score(papers: List[Dict], config: dict) -> List[Dict]:
    min_score = config.get("scoring", {}).get("min_score", 1)
    scored = []

    for paper in papers:
        s = score_paper(paper, config)
        if s >= min_score:
            paper["score"] = s
            scored.append(paper)

    # Deduplicate by normalized title — keep highest-scoring version
    best: dict = {}
    for p in scored:
        key = re.sub(r"\s+", " ", p["title"].lower().strip())
        if key not in best or p["score"] > best[key]["score"]:
            best[key] = p
    unique = list(best.values())

    unique.sort(key=lambda p: -p["score"])

    logger.info(
        f"Filter: {len(papers)} total -> {len(unique)} after scoring (min_score={min_score})"
    )
    return unique
