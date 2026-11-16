from datetime import datetime, timezone
import difflib
import re
from typing import Dict, List, Set, Tuple
import uuid

from app.logging_config import get_logger
from pipeline.sources_trust import SourceTrustManager
from schemas.common import SourceTier, SportType
from schemas.discovery import DiscoveredArticle, StoryCluster, ViralityBreakdown

logger = get_logger(__name__)

STOPWORDS = {
    "a", "an", "the", "in", "on", "at", "to", "for", "with", "and", "or", "is", "are",
    "was", "were", "be", "been", "of", "by", "from", "as", "about", "into", "over",
    "after", "vs", "against", "report", "news", "update", "latest", "says", "claims",
    "ahead", "how", "why", "what", "who", "when", "where", "which",
}


class StoryClusteringEngine:
    """Clusters articles covering the same event using entity and title similarity."""

    def __init__(
        self,
        trust_manager: SourceTrustManager,
        similarity_threshold: float = 0.45,
    ):
        self.trust_manager = trust_manager
        self.similarity_threshold = similarity_threshold

    def extract_entities(self, text: str, sport: SportType) -> List[str]:
        """Extract mentioned star entities and proper nouns from title."""
        matched: List[str] = []
        lower_text = text.lower()

        # Check known star entities
        star_entities = self.trust_manager.get_star_entities(sport)
        for entity in star_entities:
            if entity.lower() in lower_text:
                matched.append(entity)

        # Extract capitalized multi-word phrases as proper nouns
        proper_nouns = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", text)
        for pn in proper_nouns:
            if pn.lower() not in STOPWORDS and pn not in matched and len(pn) > 2:
                matched.append(pn)

        return matched

    def clean_tokens(self, text: str) -> Set[str]:
        """Tokenize and remove stopwords and punctuation."""
        words = re.findall(r"\b[a-zA-Z0-9]+\b", text.lower())
        return {w for w in words if w not in STOPWORDS and len(w) > 2}

    def compute_similarity(
        self,
        art1: DiscoveredArticle,
        art2: DiscoveredArticle,
        entities1: List[str],
        entities2: List[str],
    ) -> float:
        """Compute hybrid similarity between two articles."""
        tokens1 = self.clean_tokens(art1.title)
        tokens2 = self.clean_tokens(art2.title)

        # Token Jaccard similarity
        union = tokens1.union(tokens2)
        if not union:
            return 0.0
        jaccard = len(tokens1.intersection(tokens2)) / len(union)

        # Sequence ratio on cleaned lower strings
        clean1 = " ".join(sorted(tokens1))
        clean2 = " ".join(sorted(tokens2))
        seq_ratio = difflib.SequenceMatcher(None, clean1, clean2).ratio()

        # Entity overlap
        set_ent1 = {e.lower() for e in entities1}
        set_ent2 = {e.lower() for e in entities2}
        ent_inter = set_ent1.intersection(set_ent2)
        entity_score = 0.0
        if set_ent1 and set_ent2:
            entity_score = len(ent_inter) / max(1, min(len(set_ent1), len(set_ent2)))

        # Shared key star entity provides a strong boost
        boost = 0.2 if len(ent_inter) >= 1 else 0.0

        combined = (0.45 * jaccard) + (0.35 * seq_ratio) + (0.20 * entity_score) + boost
        return min(1.0, combined)

    def select_primary_headline(self, articles: List[DiscoveredArticle]) -> str:
        """Select best representative headline prioritizing Tier 1/2 sources and clarity."""
        tier_weights = {
            SourceTier.TIER_1: 3,
            SourceTier.TIER_2: 2,
            SourceTier.TIER_3: 1,
            SourceTier.UNKNOWN: 0,
        }

        # Sort by tier weight desc, then by length closest to 70 chars
        sorted_articles = sorted(
            articles,
            key=lambda a: (
                tier_weights.get(a.source_tier, 0),
                -abs(len(a.title) - 70),
            ),
            reverse=True,
        )
        return sorted_articles[0].title if sorted_articles else ""

    def cluster_articles(
        self,
        articles: List[DiscoveredArticle],
        sport: SportType,
    ) -> List[StoryCluster]:
        """Cluster articles covering the same event into StoryCluster objects."""
        if not articles:
            return []

        article_entities = [self.extract_entities(a.title, sport) for a in articles]
        clusters: List[List[int]] = []  # Indices of articles in each cluster

        for i, article in enumerate(articles):
            best_cluster_idx: Optional[int] = None
            best_sim = 0.0

            for c_idx, cluster in enumerate(clusters):
                # Calculate average similarity to articles in cluster
                sims = [
                    self.compute_similarity(article, articles[m_idx], article_entities[i], article_entities[m_idx])
                    for m_idx in cluster
                ]
                avg_sim = sum(sims) / len(sims)
                if avg_sim > best_sim:
                    best_sim = avg_sim
                    best_cluster_idx = c_idx

            if best_cluster_idx is not None and best_sim >= self.similarity_threshold:
                clusters[best_cluster_idx].append(i)
            else:
                clusters.append([i])

        # Form StoryCluster objects
        now = datetime.now(timezone.utc)
        result_clusters: List[StoryCluster] = []

        for cluster_indices in clusters:
            cluster_articles = [articles[idx] for idx in cluster_indices]
            # Collect unique entities
            cluster_ents: Set[str] = set()
            for idx in cluster_indices:
                cluster_ents.update(article_entities[idx])

            # Collect unique domains
            domains = list({a.domain for a in cluster_articles if a.domain})
            headline = self.select_primary_headline(cluster_articles)

            # Determine age in hours from newest article in cluster
            newest_pub = max(a.published_at for a in cluster_articles)
            if newest_pub.tzinfo is None:
                newest_pub = newest_pub.replace(tzinfo=timezone.utc)
            age_hours = max(0.0, (now - newest_pub).total_seconds() / 3600.0)

            result_clusters.append(
                StoryCluster(
                    cluster_id=str(uuid.uuid4()),
                    sport=sport,
                    primary_headline=headline,
                    articles=cluster_articles,
                    entities=sorted(cluster_ents),
                    source_count=len(cluster_articles),
                    independent_domains=domains,
                    age_hours=round(age_hours, 1),
                    virality=ViralityBreakdown(),  # Will be computed in ranking stage
                )
            )

        logger.info(
            "Clustered %d articles into %d story clusters for %s",
            len(articles),
            len(result_clusters),
            sport.value,
        )
        return result_clusters
