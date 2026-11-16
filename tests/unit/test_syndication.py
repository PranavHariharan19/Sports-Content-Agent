from pipeline.syndication import SyndicationDeduplicator
from schemas.common import SourceTier
from schemas.sources import StoredSource


SAMPLE_WIRE_COPY_1 = (
    "LONDON (Reuters) - Arsenal took a major step toward the Premier League title with a commanding "
    "3-1 victory over Chelsea at Emirates Stadium on Saturday. Bukayo Saka and Declan Rice scored first-half "
    "goals before Kai Havertz added a third in the 68th minute to seal the three points. "
    "Chelsea pulled one goal back through Nicolas Jackson but could not mount a serious comeback."
)

SAMPLE_WIRE_COPY_2 = (
    "Arsenal took a major step toward the Premier League title with a commanding 3-1 victory over Chelsea "
    "at Emirates Stadium on Saturday. Bukayo Saka and Declan Rice scored first-half goals before Kai Havertz "
    "added a third in the 68th minute to seal the three points. Chelsea pulled one goal back through Nicolas "
    "Jackson but could not mount a serious comeback against Mikel Arteta's side."
)

DISTINCT_ARTICLE = (
    "Arsenal manager Mikel Arteta praised his team's maturity following their convincing 3-1 win over Chelsea. "
    "Speaking to reporters after the match, Arteta emphasized that tactical discipline in midfield was decisive. "
    "He reserved special praise for captain Martin Odegaard, whose leadership controlled the tempo."
)


def test_syndication_detection_and_grouping():
    """Verify near-identical wire articles from different domains are grouped into 1 syndication cluster."""
    dedup = SyndicationDeduplicator(similarity_threshold=0.65)

    src1 = StoredSource(
        source_id="src_1",
        job_id="job_syn",
        url="https://reuters.com/sports/soccer/1",
        domain="reuters.com",
        source_tier=SourceTier.TIER_2,
        cleaned_text=SAMPLE_WIRE_COPY_1,
    )
    src2 = StoredSource(
        source_id="src_2",
        job_id="job_syn",
        url="https://skysports.com/football/reuters-feed/2",
        domain="skysports.com",
        source_tier=SourceTier.TIER_2,
        cleaned_text=SAMPLE_WIRE_COPY_2,
    )
    src3 = StoredSource(
        source_id="src_3",
        job_id="job_syn",
        url="https://theguardian.com/football/arteta-reaction",
        domain="theguardian.com",
        source_tier=SourceTier.TIER_2,
        cleaned_text=DISTINCT_ARTICLE,
    )

    processed = dedup.deduplicate([src1, src2, src3])

    # src1 and src2 must be syndicated together
    assert processed[0].is_syndicated is True
    assert processed[1].is_syndicated is True
    assert processed[0].syndication_cluster_id == processed[1].syndication_cluster_id

    # src3 is an independent distinct article
    assert processed[2].is_syndicated is False
    assert processed[2].syndication_cluster_id is None

    # Corroborating independent count must be 2 (the 2 syndicated copies count as 1, plus src3)
    indep_count = SyndicationDeduplicator.count_independent_corroborating_sources(processed)
    assert indep_count == 2
