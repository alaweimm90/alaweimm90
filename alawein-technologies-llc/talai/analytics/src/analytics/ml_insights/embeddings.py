"""
Embedding-based Analysis for TalAI

Semantic similarity detection, clustering, and embedding-based insights
for hypothesis and research analysis.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans, DBSCAN
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import hashlib
import pickle
from pathlib import Path

logger = logging.getLogger(__name__)

class EmbeddingProcessor:
    """
    Process and analyze text embeddings for semantic similarity and insights.

    Features:
    - Hypothesis similarity detection
    - Semantic clustering
    - Embedding-based search
    - Dimensionality reduction for visualization
    - Cache management for embeddings
    """

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', cache_dir: Optional[str] = None):
        """Initialize embedding processor."""
        self.model_name = model_name
        self.cache_dir = Path(cache_dir or '/tmp/talai_embeddings')
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.embeddings_cache = {}
        self._initialize_model()

    def _initialize_model(self):
        """Initialize the sentence transformer model."""
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            logger.info(f"Initialized {self.model_name} with {self.embedding_dim} dimensions")
        except ImportError:
            logger.warning("sentence-transformers not available, using mock embeddings")
            self.model = None
            self.embedding_dim = 384  # Default dimension

    async def compute_embeddings(self, texts: List[str], use_cache: bool = True) -> np.ndarray:
        """
        Compute embeddings for a list of texts.

        Args:
            texts: List of text strings
            use_cache: Whether to use cached embeddings

        Returns:
            Array of embeddings
        """
        try:
            embeddings = []

            for text in texts:
                # Check cache
                text_hash = hashlib.sha256(text.encode()).hexdigest()

                if use_cache and text_hash in self.embeddings_cache:
                    embeddings.append(self.embeddings_cache[text_hash])
                else:
                    # Compute embedding
                    if self.model:
                        embedding = self.model.encode(text, show_progress_bar=False)
                    else:
                        # Mock embedding for testing
                        embedding = self._generate_mock_embedding(text)

                    # Cache it
                    self.embeddings_cache[text_hash] = embedding
                    embeddings.append(embedding)

            return np.array(embeddings)

        except Exception as e:
            logger.error(f"Error computing embeddings: {e}")
            raise

    def _generate_mock_embedding(self, text: str) -> np.ndarray:
        """Generate mock embedding for testing."""
        # Create deterministic pseudo-embedding from text
        np.random.seed(hash(text) % (2**32))
        return np.random.randn(self.embedding_dim)

    async def find_similar_hypotheses(self,
                                     query_hypothesis: str,
                                     hypothesis_database: List[Dict],
                                     top_k: int = 5,
                                     threshold: float = 0.7) -> List[Dict]:
        """
        Find similar hypotheses using embedding similarity.

        Args:
            query_hypothesis: Query hypothesis text
            hypothesis_database: List of hypothesis dictionaries
            top_k: Number of similar hypotheses to return
            threshold: Minimum similarity threshold

        Returns:
            List of similar hypotheses with similarity scores
        """
        try:
            if not hypothesis_database:
                return []

            # Compute query embedding
            query_embedding = await self.compute_embeddings([query_hypothesis])

            # Compute database embeddings
            db_texts = [h.get('statement', '') for h in hypothesis_database]
            db_embeddings = await self.compute_embeddings(db_texts)

            # Calculate similarities
            similarities = cosine_similarity(query_embedding, db_embeddings)[0]

            # Get top-k similar hypotheses
            similar_indices = np.argsort(similarities)[::-1][:top_k+1]  # +1 to exclude self

            results = []
            for idx in similar_indices:
                if similarities[idx] >= threshold and db_texts[idx] != query_hypothesis:
                    hypothesis = hypothesis_database[idx].copy()
                    hypothesis['similarity_score'] = float(similarities[idx])
                    hypothesis['similarity_rank'] = len(results) + 1
                    results.append(hypothesis)

                    if len(results) >= top_k:
                        break

            return results

        except Exception as e:
            logger.error(f"Error finding similar hypotheses: {e}")
            return []

    async def cluster_hypotheses(self,
                                hypotheses: List[Dict],
                                method: str = 'kmeans',
                                n_clusters: Optional[int] = None) -> Dict[str, Any]:
        """
        Cluster hypotheses based on semantic similarity.

        Args:
            hypotheses: List of hypothesis dictionaries
            method: Clustering method ('kmeans', 'dbscan')
            n_clusters: Number of clusters (for kmeans)

        Returns:
            Clustering results with assignments and statistics
        """
        try:
            if len(hypotheses) < 2:
                return {'error': 'Need at least 2 hypotheses for clustering'}

            # Extract texts and compute embeddings
            texts = [h.get('statement', '') for h in hypotheses]
            embeddings = await self.compute_embeddings(texts)

            # Apply clustering
            if method == 'kmeans':
                if n_clusters is None:
                    # Use elbow method to determine optimal clusters
                    n_clusters = min(int(np.sqrt(len(hypotheses) / 2)), 10)

                clusterer = KMeans(n_clusters=n_clusters, random_state=42)
                labels = clusterer.fit_predict(embeddings)
                centers = clusterer.cluster_centers_

            elif method == 'dbscan':
                clusterer = DBSCAN(eps=0.3, min_samples=2, metric='cosine')
                labels = clusterer.fit_predict(embeddings)
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                centers = None

            else:
                return {'error': f'Unknown clustering method: {method}'}

            # Organize results by cluster
            clusters = {}
            for idx, label in enumerate(labels):
                if label not in clusters:
                    clusters[label] = []

                hypothesis_info = {
                    'hypothesis_id': hypotheses[idx].get('id', str(idx)),
                    'statement': texts[idx][:200],
                    'domain': hypotheses[idx].get('domain', 'unknown')
                }
                clusters[label].append(hypothesis_info)

            # Calculate cluster statistics
            cluster_stats = []
            for label, members in clusters.items():
                if label == -1:  # DBSCAN noise points
                    continue

                # Get cluster embeddings
                cluster_indices = [i for i, l in enumerate(labels) if l == label]
                cluster_embeddings = embeddings[cluster_indices]

                # Calculate intra-cluster similarity
                if len(cluster_embeddings) > 1:
                    intra_similarity = np.mean(
                        cosine_similarity(cluster_embeddings, cluster_embeddings)
                    )
                else:
                    intra_similarity = 1.0

                # Find cluster theme (most representative hypothesis)
                if centers is not None and label >= 0:
                    center = centers[label]
                    distances = cosine_similarity([center], cluster_embeddings)[0]
                    representative_idx = cluster_indices[np.argmax(distances)]
                    theme = texts[representative_idx][:100]
                else:
                    # Use centroid for DBSCAN
                    centroid = np.mean(cluster_embeddings, axis=0)
                    distances = cosine_similarity([centroid], cluster_embeddings)[0]
                    representative_idx = cluster_indices[np.argmax(distances)]
                    theme = texts[representative_idx][:100]

                cluster_stats.append({
                    'cluster_id': int(label),
                    'size': len(members),
                    'intra_cluster_similarity': float(intra_similarity),
                    'theme': theme,
                    'domains': list(set(m['domain'] for m in members))
                })

            # Sort clusters by size
            cluster_stats.sort(key=lambda x: x['size'], reverse=True)

            return {
                'method': method,
                'n_clusters': n_clusters,
                'clusters': clusters,
                'cluster_statistics': cluster_stats,
                'noise_points': len(clusters.get(-1, [])) if method == 'dbscan' else 0,
                'silhouette_score': await self._calculate_silhouette_score(embeddings, labels)
            }

        except Exception as e:
            logger.error(f"Error clustering hypotheses: {e}")
            return {'error': str(e)}

    async def _calculate_silhouette_score(self,
                                         embeddings: np.ndarray,
                                         labels: np.ndarray) -> float:
        """Calculate silhouette score for clustering quality."""
        try:
            from sklearn.metrics import silhouette_score

            # Filter out noise points for DBSCAN
            mask = labels != -1
            if np.sum(mask) < 2:
                return 0.0

            score = silhouette_score(embeddings[mask], labels[mask], metric='cosine')
            return float(score)
        except:
            return 0.0

    async def generate_embedding_visualization(self,
                                              hypotheses: List[Dict],
                                              method: str = 'tsne') -> Dict[str, Any]:
        """
        Generate 2D visualization coordinates for embeddings.

        Args:
            hypotheses: List of hypothesis dictionaries
            method: Dimensionality reduction method ('tsne', 'pca')

        Returns:
            2D coordinates and metadata for visualization
        """
        try:
            if len(hypotheses) < 2:
                return {'error': 'Need at least 2 hypotheses for visualization'}

            # Compute embeddings
            texts = [h.get('statement', '') for h in hypotheses]
            embeddings = await self.compute_embeddings(texts)

            # Apply dimensionality reduction
            if method == 'tsne':
                reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(hypotheses)-1))
                coords_2d = reducer.fit_transform(embeddings)
            elif method == 'pca':
                reducer = PCA(n_components=2, random_state=42)
                coords_2d = reducer.fit_transform(embeddings)
            else:
                return {'error': f'Unknown method: {method}'}

            # Prepare visualization data
            viz_data = []
            for idx, (x, y) in enumerate(coords_2d):
                viz_data.append({
                    'x': float(x),
                    'y': float(y),
                    'hypothesis_id': hypotheses[idx].get('id', str(idx)),
                    'statement': texts[idx][:100],
                    'domain': hypotheses[idx].get('domain', 'unknown'),
                    'success_rate': hypotheses[idx].get('success_rate', 0.5)
                })

            # Calculate clusters for coloring
            if len(hypotheses) > 5:
                kmeans = KMeans(n_clusters=min(5, len(hypotheses)//3), random_state=42)
                cluster_labels = kmeans.fit_predict(embeddings)
                for i, label in enumerate(cluster_labels):
                    viz_data[i]['cluster'] = int(label)

            return {
                'method': method,
                'coordinates': viz_data,
                'n_points': len(viz_data),
                'explained_variance': float(reducer.explained_variance_ratio_.sum())
                                     if method == 'pca' else None
            }

        except Exception as e:
            logger.error(f"Error generating visualization: {e}")
            return {'error': str(e)}

    async def semantic_search(self,
                            query: str,
                            documents: List[Dict],
                            top_k: int = 10,
                            filters: Optional[Dict] = None) -> List[Dict]:
        """
        Perform semantic search over documents.

        Args:
            query: Search query
            documents: List of document dictionaries
            top_k: Number of results to return
            filters: Optional filters to apply

        Returns:
            Ranked search results
        """
        try:
            if not documents:
                return []

            # Apply filters if provided
            if filters:
                filtered_docs = []
                for doc in documents:
                    match = True
                    for key, value in filters.items():
                        if doc.get(key) != value:
                            match = False
                            break
                    if match:
                        filtered_docs.append(doc)
                documents = filtered_docs

            if not documents:
                return []

            # Compute query embedding
            query_embedding = await self.compute_embeddings([query])

            # Compute document embeddings
            doc_texts = [self._extract_searchable_text(doc) for doc in documents]
            doc_embeddings = await self.compute_embeddings(doc_texts)

            # Calculate similarities
            similarities = cosine_similarity(query_embedding, doc_embeddings)[0]

            # Rank and return top results
            ranked_indices = np.argsort(similarities)[::-1][:top_k]

            results = []
            for rank, idx in enumerate(ranked_indices):
                if similarities[idx] > 0:  # Only include positive similarities
                    result = documents[idx].copy()
                    result['relevance_score'] = float(similarities[idx])
                    result['search_rank'] = rank + 1
                    results.append(result)

            return results

        except Exception as e:
            logger.error(f"Error in semantic search: {e}")
            return []

    def _extract_searchable_text(self, document: Dict) -> str:
        """Extract searchable text from document."""
        parts = []

        # Add main text fields
        for field in ['statement', 'description', 'abstract', 'content', 'text']:
            if field in document and document[field]:
                parts.append(str(document[field]))

        # Add metadata
        if document.get('domain'):
            parts.append(f"Domain: {document['domain']}")

        if document.get('keywords'):
            parts.append(f"Keywords: {', '.join(document['keywords'])}")

        return ' '.join(parts)[:1000]  # Limit length

    async def detect_duplicate_content(self,
                                      documents: List[Dict],
                                      threshold: float = 0.95) -> List[List[int]]:
        """
        Detect near-duplicate content using embeddings.

        Args:
            documents: List of documents
            threshold: Similarity threshold for duplicates

        Returns:
            Groups of duplicate document indices
        """
        try:
            if len(documents) < 2:
                return []

            # Compute embeddings
            texts = [self._extract_searchable_text(doc) for doc in documents]
            embeddings = await self.compute_embeddings(texts)

            # Compute pairwise similarities
            similarities = cosine_similarity(embeddings)

            # Find duplicate groups
            duplicate_groups = []
            processed = set()

            for i in range(len(documents)):
                if i in processed:
                    continue

                # Find all documents similar to this one
                similar = [j for j in range(len(documents))
                          if j != i and similarities[i][j] >= threshold]

                if similar:
                    group = [i] + similar
                    duplicate_groups.append(group)
                    processed.update(group)

            return duplicate_groups

        except Exception as e:
            logger.error(f"Error detecting duplicates: {e}")
            return []

    async def save_embeddings_cache(self):
        """Save embeddings cache to disk."""
        try:
            cache_file = self.cache_dir / f"{self.model_name}_cache.pkl"
            with open(cache_file, 'wb') as f:
                pickle.dump(self.embeddings_cache, f)
            logger.info(f"Saved {len(self.embeddings_cache)} embeddings to cache")
        except Exception as e:
            logger.error(f"Error saving cache: {e}")

    async def load_embeddings_cache(self):
        """Load embeddings cache from disk."""
        try:
            cache_file = self.cache_dir / f"{self.model_name}_cache.pkl"
            if cache_file.exists():
                with open(cache_file, 'rb') as f:
                    self.embeddings_cache = pickle.load(f)
                logger.info(f"Loaded {len(self.embeddings_cache)} embeddings from cache")
        except Exception as e:
            logger.error(f"Error loading cache: {e}")