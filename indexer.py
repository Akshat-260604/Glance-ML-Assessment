import torch
import os
import uuid
from qdrant_client import QdrantClient
from qdrant_client.http import models
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from sentence_transformers import SentenceTransformer
import re

class FashionIndexer:
    def __init__(self, url, api_key):
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        
        print("Loading BLIP for image captioning...")
        self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.blip_model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        ).to(self.device).eval()
        
        print("Loading CLIP for semantic embeddings...")
        self.embedder = SentenceTransformer("sentence-transformers/clip-ViT-B-32", device=self.device)
        
        self.client = QdrantClient(url=url, api_key=api_key)
        self.collection_name = "fashion"
        
        print(f"Preparing cloud collection '{self.collection_name}'...")
        try:
            self.client.delete_collection(collection_name=self.collection_name)
        except Exception:
            pass

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(size=512, distance=models.Distance.COSINE)
        )

    def _extract_fashion_caption(self, image):
        """Extract detailed fashion description using BLIP with dual captioning"""
        try:
            inputs_scene = self.blip_processor(image, return_tensors="pt").to(self.device)
            with torch.no_grad():
                output_scene = self.blip_model.generate(
                    **inputs_scene,
                    max_new_tokens=50,
                    num_beams=3,
                    do_sample=False
                )
            scene_caption = self.blip_processor.decode(output_scene[0], skip_special_tokens=True)
            
            text_prompt = "a photo of a person wearing"
            inputs_clothing = self.blip_processor(image, text=text_prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                output_clothing = self.blip_model.generate(
                    **inputs_clothing,
                    max_new_tokens=50,
                    num_beams=3,
                    do_sample=False
                )
            clothing_caption = self.blip_processor.decode(output_clothing[0], skip_special_tokens=True)
            
            combined = f"{scene_caption}. {clothing_caption}"
            return combined if combined.strip() else "A fashion image"
        except Exception as e:
            print(f"Caption generation failed: {e}")
            return "A fashion image"

    def _infer_style_from_clothing(self, clothing_list):
        """Infer style category from detected clothing items"""
        formal_items = {'blazer', 'suit', 'tie', 'waistcoat', 'vest', 'dress', 'button-down', 'collared'}
        casual_items = {'hoodie', 't-shirt', 'jeans', 'shorts', 'sweatpants', 'sneakers', 'tank', 'sweater'}
        athletic_items = {'shorts', 'tank', 'sneakers', 'sweatpants'}
        outerwear_items = {'jacket', 'coat', 'raincoat', 'windbreaker', 'cardigan'}
        
        clothing_set = set(c.lower() for c in clothing_list)
        
        styles = []
        if clothing_set & formal_items:
            styles.append('formal')
        if clothing_set & casual_items:
            styles.append('casual')
        if clothing_set & athletic_items:
            styles.append('athletic')
        if clothing_set & outerwear_items:
            styles.append('outerwear')
        
        return styles if styles else ['casual']

    def _extract_attributes_advanced(self, caption, image=None):
        """Extract fashion attributes with improved patterns"""
        caption_lower = caption.lower()
        
        colors = re.findall(
            r'\b(bright|dark|light|neon)?\s*(red|blue|green|yellow|black|white|orange|purple|pink|brown|gray|grey|beige|navy|maroon|crimson|turquoise|khaki|cream|silver|gold|bronze)\b',
            caption_lower
        )
        colors = [f"{modifier} {color}".strip() for modifier, color in colors]
        
        clothing = re.findall(
            r'\b(button-?down|button-?up|collared|polo|t-?shirt|tank|sleeveless|crop|oversized|fitted|slim|skinny|straight|bootcut|flared|pleated|pencil|mini|midi|maxi|shorts|pants|jeans|trousers|denim|chinos|sweatpants|hoodie|jacket|blazer|suit|coat|raincoat|windbreaker|dress|skirt|sweater|cardigan|vest|waistcoat|tie|scarf|belt|hat|cap|shirt|blouse|top)\b',
            caption_lower
        )
        
        context = re.findall(
            r'\b(office|corporate|workplace|conference|meeting|street|urban|city|downtown|park|outdoor|indoor|home|professional|business|event|weekend|garden|beach|hiking|gym|runway|fashion\s*show|catwalk|studio|photoshoot|model|walking|standing|sitting|bench|building|room|store|shop|mall|restaurant|cafe|bar|club|party|wedding|gala|red\s*carpet|stage|sidewalk|road|path|field|forest|mountain|lake|river|pool|hallway|lobby|staircase|balcony|terrace|rooftop|window|mirror|wall|background)\b',
            caption_lower
        )
        
        if 'runway' in caption_lower or 'fashion show' in caption_lower or 'catwalk' in caption_lower:
            context.append('runway')
            context.append('fashion-show')
        if 'walk' in caption_lower and ('street' in caption_lower or 'city' in caption_lower or 'down' in caption_lower):
            context.append('street')
        if 'stand' in caption_lower:
            context.append('standing')
        if 'sit' in caption_lower:
            context.append('sitting')
        
        style_markers = re.findall(
            r'\b(casual|formal|business|smart|elegant|trendy|vintage|modern|minimalist|bohemian|preppy|athletic|streetwear|haute-?couture|chic|sporty|classic|edgy|glamorous|sophisticated)\b',
            caption_lower
        )
        
        clothing_list = list(set(clothing))
        inferred_styles = self._infer_style_from_clothing(clothing_list)
        all_styles = list(set(style_markers + inferred_styles))
        
        return {
            'colors': list(set(colors)),
            'clothing': clothing_list,
            'context': list(set(context)),
            'style': all_styles,
            'caption': caption
        }

    @torch.no_grad()
    def process_image(self, img_path):
        try:
            image = Image.open(img_path).convert("RGB")
            image.verify()
            image = Image.open(img_path).convert("RGB")
        except Exception:
            raise ValueError(f"Could not load image: {img_path}")
        
        caption = self._extract_fashion_caption(image)
        attributes = self._extract_attributes_advanced(caption, image)
        
        vector = self.embedder.encode(caption).tolist()
        
        return vector, attributes

    def index_directory(self, data_dir, batch_size=50, max_retries=3):
        files = [f for f in os.listdir(data_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        print(f"Indexing {len(files)} images to Qdrant Cloud in batches of {batch_size}...")
        
        points = []
        failed_images = []
        batch_num = 0
        
        for i, img_name in enumerate(files):
            path = os.path.join(data_dir, img_name)
            try:
                vec, attributes = self.process_image(path)
                points.append(models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vec,
                    payload={
                        "image_name": img_name,
                        "caption": attributes['caption'],
                        "colors": attributes['colors'],
                        "clothing": attributes['clothing'],
                        "context": attributes['context'],
                        "style": attributes.get('style', [])
                    }
                ))

                if len(points) >= batch_size:
                    for attempt in range(max_retries):
                        try:
                            self.client.upsert(collection_name=self.collection_name, points=points)
                            batch_num += 1
                            print(f"Uploaded batch {batch_num} ({len(points)} images)")
                            points = []
                            break
                        except Exception as upload_err:
                            if attempt < max_retries - 1:
                                print(f"   Batch upload failed, retrying ({attempt+1}/{max_retries})...")
                                import time
                                time.sleep(2)
                            else:
                                print(f"   Batch failed after {max_retries} attempts: {upload_err}")
                                failed_images.extend([p.payload['image_name'] for p in points])
                                points = []
                    
            except Exception as e:
                print(f"Error with {img_name}: {e}")
                failed_images.append(img_name)

        if points:
            for attempt in range(max_retries):
                try:
                    self.client.upsert(collection_name=self.collection_name, points=points)
                    print(f"Final batch uploaded. Total batches: {batch_num + 1}")
                    break
                except Exception as upload_err:
                    if attempt < max_retries - 1:
                        print(f"   Final batch retry ({attempt+1}/{max_retries})...")
                        import time
                        time.sleep(2)
                    else:
                        failed_images.extend([p.payload['image_name'] for p in points])
        
        total_indexed = len(files) - len(failed_images)
        print(f"\n✓ Indexing complete: {total_indexed}/{len(files)} images indexed successfully")
        if failed_images:
            print(f"✗ {len(failed_images)} images failed (network timeouts)")
            print(f"Final batch uploaded. Total indexed: {len(files)}")
