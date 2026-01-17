import torch
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import numpy as np

class FashionRetriever:
    def __init__(self, url, api_key):
        self.client = QdrantClient(url=url, api_key=api_key)
        self.collection_name = "fashion"
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

        print("Loading CLIP-based SentenceTransformer for retrieval...")
        self.embedder = SentenceTransformer("sentence-transformers/clip-ViT-B-32", device=self.device)

    def _compute_composition_score(self, query, caption):
        """
        Compute composition accuracy: do color-item pairs match in correct order?
        Example: "red pants black jacket" vs "red jacket black pants"
        """
        query_lower = query.lower()
        caption_lower = caption.lower()
        
        colors = ['red', 'blue', 'green', 'yellow', 'black', 'white', 'orange', 
                  'purple', 'pink', 'brown', 'gray', 'grey', 'beige', 'navy']
        items = ['pants', 'jacket', 'shirt', 'dress', 'skirt', 'coat', 'top', 
                 'jeans', 'shorts', 'sweater', 'hoodie', 'blazer', 'suit']
        
        query_pairs = []
        for color in colors:
            for item in items:
                if f"{color} {item}" in query_lower or f"{color}.*{item}" in query_lower:
                    query_pairs.append((color, item))
        
        if not query_pairs:
            return 1.0
        
        match_score = 0.0
        for color, item in query_pairs:
            if f"{color} {item}" in caption_lower or f"{color}.*{item}" in caption_lower:
                match_score += 1.0
            elif color in caption_lower and item in caption_lower:
                match_score += 0.3
        
        return match_score / len(query_pairs)

    def _compute_intelligent_rerank(self, results, query):
        """
        Intelligent re-ranking using:
        1. CLIP embedding similarity (base score from Qdrant)
        2. Direct caption-query embedding similarity  
        3. Composition accuracy (color-item pair matching)
        4. Caption uniqueness (longer/more detailed = better)
        5. Inter-result diversity (MMR algorithm)
        
        Returns scores in natural [0, 1] range from CLIP similarities.
        """
        if not results:
            return results
        
        query_embedding = self.embedder.encode(query, convert_to_tensor=True)
        
        captions = [r.payload.get('caption', '') for r in results]
        caption_embeddings = self.embedder.encode(captions, convert_to_tensor=True)
        
        direct_similarity = torch.nn.functional.cosine_similarity(
            query_embedding.unsqueeze(0), 
            caption_embeddings, 
            dim=1
        ).cpu().numpy()
        
        composition_scores = np.array([
            self._compute_composition_score(query, caption) 
            for caption in captions
        ])
        
        caption_lengths = np.array([len(c.split()) for c in captions])
        avg_length = caption_lengths.mean()
        length_deviation = (caption_lengths - avg_length) / (avg_length + 1e-6)
        uniqueness_scores = 1.0 / (1.0 + np.exp(-length_deviation * 2))
        
        diversity_scores = []
        selected_embeddings = []
        
        for i, emb in enumerate(caption_embeddings):
            if not selected_embeddings:
                diversity_scores.append(1.0)
            else:
                similarities = torch.stack([
                    torch.nn.functional.cosine_similarity(
                        emb.unsqueeze(0), 
                        sel_emb.unsqueeze(0), 
                        dim=1
                    )
                    for sel_emb in selected_embeddings
                ])
                max_sim = similarities.max().item()
                diversity_scores.append(1.0 - max_sim * 0.3)
            
            if i < len(results) // 2:
                selected_embeddings.append(emb)
        
        diversity_scores = np.array(diversity_scores)
        
        base_scores = np.array([r.score for r in results])
        
        final_scores = (
            base_scores * 0.40 +
            direct_similarity * 0.30 +
            composition_scores * 0.20 +
            (uniqueness_scores - 0.5) * 0.05 +
            (diversity_scores - 1.0) * 0.05
        )
        
        final_scores = np.clip(final_scores, 0.0, 1.0)
        
        for i, result in enumerate(results):
            result.score = float(final_scores[i])
        
        return sorted(results, key=lambda x: x.score, reverse=True)

    def search(self, query, k=3):
        """
        Intelligent semantic search with automatic re-ranking.
        Uses CLIP embeddings to compute relevance, uniqueness, and diversity.
        
        Args:
            query: Natural language description
            k: Number of results to return
            
        Returns:
            List of top-k results with intelligently computed scores
        """
        try:
            query_vec = self.embedder.encode(query).tolist()
        except Exception as e:
            print(f"Encoding error: {e}")
            return []

        try:
            raw_results = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vec,
                limit=k * 5,
                with_payload=True
            ).points
        except Exception as e:
            print(f"Search error: {e}")
            return []

        if not raw_results:
            return []

        reranked_results = self._compute_intelligent_rerank(raw_results, query)

        seen_images = set()
        unique_results = []
        for result in reranked_results:
            img_name = result.payload.get('image_name', '')
            if img_name not in seen_images:
                seen_images.add(img_name)
                unique_results.append(result)
                if len(unique_results) >= k:
                    break

        return unique_results