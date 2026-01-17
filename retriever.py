import torch
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import re

class FashionRetriever:
    def __init__(self, url, api_key):
        self.client = QdrantClient(url=url, api_key=api_key)
        self.collection_name = "fashion"
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

        print("Loading CLIP-based SentenceTransformer for retrieval...")
        self.embedder = SentenceTransformer("sentence-transformers/clip-ViT-B-32", device=self.device)

    def _parse_query_attributes(self, query):
        query_lower = query.lower()
        
        colors = re.findall(
            r'\b(bright|dark|light|neon)?\s*(red|blue|green|yellow|black|white|orange|purple|pink|brown|gray|grey|beige|navy|maroon|crimson|turquoise|khaki|cream|silver|gold|bronze)\b',
            query_lower
        )
        colors = [f"{modifier} {color}".strip() for modifier, color in colors]
        
        clothing = re.findall(
            r'\b(button-?down|button-?up|collared|polo|t-?shirt|tank|sleeveless|crop|oversized|fitted|slim|skinny|straight|bootcut|flared|pleated|pencil|mini|midi|maxi|shorts|pants|jeans|trousers|denim|chinos|sweatpants|hoodie|jacket|blazer|suit|coat|raincoat|windbreaker|dress|skirt|sweater|cardigan|vest|waistcoat|tie|scarf|belt|hat|cap)\b',
            query_lower
        )
        
        context = re.findall(
            r'\b(office|corporate|workplace|conference|meeting|street|urban|city|downtown|park|outdoor|indoor|home|casual|formal|professional|business|event|weekend|smart-?casual|black-?tie|white-?tie|business-?casual|cocktail|garden|beach|hiking|gym)\b',
            query_lower
        )
        
        style = re.findall(
            r'\b(casual|formal|business|smart|elegant|trendy|vintage|modern|minimalist|bohemian|preppy|athletic|streetwear|haute-?couture)\b',
            query_lower
        )
        
        return {
            'colors': list(set(colors)),
            'clothing': list(set(clothing)),
            'context': list(set(context)),
            'style': list(set(style))
        }

    def _compute_attribute_boost(self, result_payload, query_attrs):
        boost_score = 0.0
        
        result_colors = set(c.lower() for c in result_payload.get('colors', []))
        result_clothing = set(c.lower() for c in result_payload.get('clothing', []))
        result_context = set(c.lower() for c in result_payload.get('context', []))
        result_style = set(s.lower() for s in result_payload.get('style', []))
        
        for color in query_attrs['colors']:
            if color.lower() in result_colors:
                boost_score += 0.20
        
        for item in query_attrs['clothing']:
            if item.lower() in result_clothing:
                boost_score += 0.25
        
        for ctx in query_attrs['context']:
            if ctx.lower() in result_context:
                boost_score += 0.15
        
        for sty in query_attrs['style']:
            if sty.lower() in result_style:
                boost_score += 0.12
        
        return boost_score

    def search(self, query, k=3):
        """
        Search for fashion images matching the query.
        
        Args:
            query: Natural language description
            k: Number of results to return
            
        Returns:
            List of top-k results with normalized scores [0, 1]
        """
        try:
            query_vec = self.embedder.encode(query).tolist()
        except Exception as e:
            print(f"Encoding error: {e}")
            return []

        try:
            from qdrant_client.models import PointStruct
            raw_results = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vec,
                limit=k * 3,
                with_payload=True
            ).points
        except Exception as e:
            print(f"Search error: {e}")
            return []

        if not raw_results:
            return []

        query_attrs = self._parse_query_attributes(query)
        
        for result in raw_results:
            attribute_boost = self._compute_attribute_boost(result.payload, query_attrs)
            normalized_boost = min(attribute_boost / 14.4, 0.5)
            
            result.score = result.score + normalized_boost

        seen_images = set()
        unique_results = []
        for result in sorted(raw_results, key=lambda x: x.score, reverse=True):
            img_name = result.payload.get('image_name', '')
            if img_name not in seen_images:
                seen_images.add(img_name)
                unique_results.append(result)
                if len(unique_results) >= k:
                    break

        if unique_results:
            max_score = max(r.score for r in unique_results)
            if max_score > 1.0:
                for result in unique_results:
                    result.score = result.score / max_score

        return unique_results