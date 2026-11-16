import difflib
import hashlib
import re
from typing import Dict, List, Set, Tuple

from app.logging_config import get_logger
from schemas.sources import StoredSource

logger = get_logger(__name__)


class SyndicationDeduplicator:
    """Detects syndicated wire copy (Reuters, AP, PA, shared PR) across publisher domains.

    Syndicated articles count as ONE source for corroboration.
    """

    def __init__(self, similarity_threshold: float = 0.70):
        self.similarity_threshold = similarity_threshold

    def create_shingles(self, text: str, k: int = 4) -> Set[str]:
        """Generate k-gram word shingles for fast near-duplicate matching."""
        words = re.findall(r"\b[a-zA-Z0-9]+\b", text.lower())
        if len(words) < k:
            return {" ".join(words)}
        return {" ".join(words[i : i + k]) for i in range(len(words) - k + 1)}

    def compute_similarity(self, text1: str, text2: str) -> float:
        """Compute shingle Jaccard similarity between two article texts."""
        if not text1 or not text2:
            return 0.0

        shingles1 = self.create_shingles(text1)
        shingles2 = self.create_shingles(text2)

        union = shingles1.union(shingles2)
        if not union:
            return 0.0

        jaccard = len(shingles1.intersection(shingles2)) / len(union)

        # If jaccard is borderline (0.6 - 0.75), verify with difflib sample
        if 0.55 <= jaccard <= 0.75:
            sample1 = " ".join(text1.split()[:200])
            sample2 = " ".join(text2.split()[:200])
            seq_ratio = difflib.SequenceMatcher(None, sample1, sample2).ratio()
            return max(jaccard, seq_ratio)

        return jaccard

    def deduplicate(self, sources: List[StoredSource]) -> List[StoredSource]:
        """Detect syndicated copies among sources and assign shared syndication_cluster_ids."""
        if len(sources) <= 1:
            return sources

        n = len(sources)
        clusters: List[List[int]] = []  # Indices of sources in each syndication cluster

        for i in range(n):
            matched_cluster: int = -1
            for c_idx, cluster in enumerate(clusters):
                # Check similarity against members of cluster
                sims = [
                    self.compute_similarity(sources[i].cleaned_text, sources[m_idx].cleaned_text)
                    for m_idx in cluster
                ]
                if any(s >= self.similarity_threshold for s in sims):
                    matched_cluster = c_idx
                    break

            if matched_cluster != -1:
                clusters[matched_cluster].append(i)
            else:
                clusters.append([i])

        # Assign syndication metadata to sources
        for cluster in clusters:
            if len(cluster) > 1:
                # Deterministic cluster ID from the sorted URLs
                cluster_urls = sorted(sources[idx].url for idx in cluster)
                syn_id = f"syn_{hashlib.sha256(';'.join(cluster_urls).encode('utf-8')).hexdigest()[:10]}"

                for idx in cluster:
                    sources[idx].is_syndicated = True
                    sources[idx].syndication_cluster_id = syn_id
                logger.info(
                    "Identified syndication cluster %s across %d sources: %s",
                    syn_id,
                    len(cluster),
                    [sources[idx].domain for idx in cluster],
                )
            else:
                sources[cluster[0]].is_syndicated = False
                sources[cluster[0]].syndication_cluster_id = None

        return sources

    @staticmethod
    def count_independent_corroborating_sources(sources: List[StoredSource]) -> int:
        """Calculate the true number of independent sources after syndication de-duplication.

        Only allowlisted sources can contribute.
        Syndicated articles sharing a syndication_cluster_id count as ONE independent source.
        """
        allowlisted = [s for s in sources if s.is_allowlisted]
        seen_syndication_groups: Set[str] = set()
        independent_count = 0

        for s in allowlisted:
            if s.is_syndicated and s.syndication_cluster_id:
                if s.syndication_cluster_id not in seen_syndication_groups:
                    seen_syndication_groups.add(s.syndication_cluster_id)
                    independent_count += 1
            else:
                independent_count += 1

        return independent_count
