"""
Tests for parsers/keyword_filter.py
Run: python3 -m pytest tests/ -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from parsers.keyword_filter import (
    score_paper,
    filter_and_score,
    _venue_bonus,
    _focus_bonus,
    _recency_bonus,
)
from datetime import datetime

CURRENT_YEAR = datetime.today().year

CONFIG = {
    "keywords": {
        "primary": [
            "CTR prediction", "CVR prediction", "click-through rate",
            "conversion rate", "click prediction", "purchase prediction",
        ],
        "secondary": [
            "recommender system", "recommendation system",
            "collaborative filtering", "sequential recommendation",
            "user behavior modeling", "feature interaction",
            "multi-task learning", "ranking model",
            "deep interest network", "industrial recommendation",
        ],
    },
    "scoring": {"min_score": 1},
}


def make_paper(**kwargs):
    defaults = {
        "title": "",
        "abstract": "",
        "venue": "arXiv",
        "source": "arxiv",
        "year": CURRENT_YEAR - 1,
        "date": f"{CURRENT_YEAR - 1}-06-01",
    }
    defaults.update(kwargs)
    return defaults


# ---------------------------------------------------------------------------
# Venue bonus
# ---------------------------------------------------------------------------

class TestVenueBonus:
    def test_recsys(self):
        assert _venue_bonus(make_paper(venue="RecSys")) == 10

    def test_kdd(self):
        assert _venue_bonus(make_paper(venue="KDD")) == 10

    def test_www(self):
        assert _venue_bonus(make_paper(venue="WWW")) == 10

    def test_sigir(self):
        assert _venue_bonus(make_paper(venue="SIGIR")) == 10

    def test_arxiv_no_bonus(self):
        assert _venue_bonus(make_paper(venue="arXiv")) == 0

    def test_case_insensitive(self):
        assert _venue_bonus(make_paper(venue="recsys")) == 10

    def test_partial_venue_name(self):
        # "ACM RecSys 2024" should still match
        assert _venue_bonus(make_paper(venue="ACM RecSys 2024")) == 10


# ---------------------------------------------------------------------------
# CTR/CVR focus bonus
# ---------------------------------------------------------------------------

class TestFocusBonus:
    def test_no_bonus_below_threshold(self):
        # 2 hits -> no bonus
        assert _focus_bonus("ctr prediction model", "a study on ctr prediction") == 0

    def test_tier1_bonus_at_3_hits(self):
        # 3 hits -> +5
        title = "ctr prediction"
        abstract = "ctr prediction click-through rate model"
        assert _focus_bonus(title, abstract) == 5

    def test_tier2_bonus_at_5_hits(self):
        # 5 hits -> +10
        title = "ctr prediction cvr prediction"
        abstract = "ctr prediction click-through rate conversion rate study"
        assert _focus_bonus(title, abstract) == 10

    def test_no_hits(self):
        assert _focus_bonus("graph neural network", "node embedding approach") == 0


# ---------------------------------------------------------------------------
# Recency bonus
# ---------------------------------------------------------------------------

class TestRecencyBonus:
    def test_current_year(self):
        assert _recency_bonus(make_paper(year=CURRENT_YEAR)) == 3

    def test_last_year(self):
        assert _recency_bonus(make_paper(year=CURRENT_YEAR - 1)) == 2

    def test_two_years_ago(self):
        assert _recency_bonus(make_paper(year=CURRENT_YEAR - 2)) == 1

    def test_older_no_bonus(self):
        assert _recency_bonus(make_paper(year=CURRENT_YEAR - 3)) == 0

    def test_missing_year(self):
        assert _recency_bonus(make_paper(year=None)) == 0


# ---------------------------------------------------------------------------
# Full score_paper integration
# ---------------------------------------------------------------------------

class TestScorePaper:
    def test_primary_keyword_in_title(self):
        paper = make_paper(title="CTR Prediction with Deep Learning")
        score = score_paper(paper, CONFIG)
        assert score >= 5  # at least +5 for primary in title

    def test_primary_keyword_in_abstract(self):
        paper = make_paper(abstract="We propose a method for CTR prediction.")
        score = score_paper(paper, CONFIG)
        assert score >= 3  # at least +3 for primary in abstract

    def test_secondary_keyword_in_title(self):
        paper = make_paper(title="A Recommender System for E-commerce")
        score = score_paper(paper, CONFIG)
        assert score >= 2

    def test_conference_paper_scores_higher_than_arxiv(self):
        base = dict(
            title="CTR Prediction via Feature Interaction",
            abstract="We study click-through rate prediction.",
            year=CURRENT_YEAR - 1,
            date=f"{CURRENT_YEAR - 1}-01-01",
        )
        conf_paper = make_paper(**base, venue="RecSys", source="conference")
        arxiv_paper = make_paper(**base, venue="arXiv", source="arxiv")
        assert score_paper(conf_paper, CONFIG) > score_paper(arxiv_paper, CONFIG)

    def test_newer_paper_scores_higher(self):
        base = dict(
            title="CTR Prediction",
            abstract="click-through rate study",
            venue="arXiv",
            source="arxiv",
        )
        newer = make_paper(**base, year=CURRENT_YEAR, date=f"{CURRENT_YEAR}-01-01")
        older = make_paper(**base, year=CURRENT_YEAR - 3, date=f"{CURRENT_YEAR - 3}-01-01")
        assert score_paper(newer, CONFIG) > score_paper(older, CONFIG)

    def test_ctr_focused_paper_gets_focus_bonus(self):
        # Many CTR/CVR mentions -> focus bonus kicks in
        focused = make_paper(
            title="CTR Prediction and CVR Prediction",
            abstract="click-through rate conversion rate ctr prediction study",
        )
        unfocused = make_paper(
            title="CTR Prediction",
            abstract="a general recommendation approach",
        )
        assert score_paper(focused, CONFIG) > score_paper(unfocused, CONFIG)

    def test_top_conference_ctr_paper_highest_score(self):
        # Best case: top conference + CTR focused + recent
        paper = make_paper(
            title="CTR Prediction with CVR Multi-task Learning",
            abstract=(
                "click-through rate prediction and conversion rate prediction "
                "are key tasks in recommender systems. We propose a ctr prediction "
                "model for industrial recommendation."
            ),
            venue="KDD",
            source="conference",
            year=CURRENT_YEAR,
        )
        score = score_paper(paper, CONFIG)
        # Should include: keyword + venue(+10) + focus(+10) + recency(+3)
        assert score >= 30

    def test_unrelated_paper_scores_zero(self):
        paper = make_paper(
            title="Image Segmentation with Transformers",
            abstract="We propose a vision transformer for semantic segmentation.",
        )
        assert score_paper(paper, CONFIG) == 0


# ---------------------------------------------------------------------------
# filter_and_score
# ---------------------------------------------------------------------------

class TestFilterAndScore:
    def test_filters_below_min_score(self):
        papers = [
            make_paper(title="Image Classification"),  # score 0
            make_paper(title="CTR Prediction Model"),  # score > 0
        ]
        result = filter_and_score(papers, CONFIG)
        assert len(result) == 1
        assert "CTR" in result[0]["title"]

    def test_deduplication(self):
        papers = [
            make_paper(title="CTR Prediction Model"),
            make_paper(title="CTR Prediction Model"),  # duplicate
            make_paper(title="ctr prediction model"),  # same, different case
        ]
        result = filter_and_score(papers, CONFIG)
        assert len(result) == 1

    def test_sorted_by_score_descending(self):
        papers = [
            make_paper(title="Recommender System Design", venue="arXiv"),
            make_paper(title="CTR Prediction via Deep Learning", venue="KDD",
                       source="conference", year=CURRENT_YEAR),
        ]
        result = filter_and_score(papers, CONFIG)
        assert result[0]["score"] >= result[-1]["score"]

    def test_conference_paper_ranks_above_arxiv_equal_content(self):
        base = dict(title="CTR Prediction Study", abstract="click-through rate model")
        papers = [
            make_paper(**base, venue="arXiv", source="arxiv"),
            make_paper(**base, venue="RecSys", source="conference"),
        ]
        result = filter_and_score(papers, CONFIG)
        assert result[0]["venue"] == "RecSys"

    def test_scores_stored_on_paper(self):
        papers = [make_paper(title="CTR Prediction")]
        result = filter_and_score(papers, CONFIG)
        assert "score" in result[0]
        assert result[0]["score"] > 0
