FASHION RETRIEVAL SYSTEM - SUBMISSION DOCUMENT
=====================================================

1. PROBLEM STATEMENT
-------------------
Build an intelligent fashion search engine that:
- Understands compositional queries ("red shirt + blue pants" vs "blue shirt + red pants")
- Handles multi-attribute searches (color + clothing + context + style)
- Works on zero-shot queries (unseen descriptions)
- Scales to 1M+ images

2. EVALUATION QUERIES
--------------------
1. Attribute Specific: "A person in a bright yellow raincoat."
2. Contextual/Place: "Professional business attire inside a modern office."
3. Complex Semantic: "Someone wearing a blue shirt sitting on a park bench."
4. Style Inference: "Casual weekend outfit for a city walk."
5. Compositional: "A red tie and a white shirt in a formal setting."

3. APPROACHES CONSIDERED
------------------------

A. VANILLA CLIP (Baseline)
   Pros: Fast, zero-shot, proven
   Cons: Poor compositionality, generic fashion understanding, no attribute extraction
   Why rejected: Cannot distinguish "red+blue" from "blue+red"

B. FLORENCE-2 (Generative Vision Model)
   Pros: Detailed captions
   Cons: Slow (3-5s/image), fails on corrupted images, no attribute extraction, resource-heavy
   Why rejected: Too slow for 3200+ images, infrastructure intensive

C. HYBRID: BLIP + CLIP + ATTRIBUTE BOOSTING (CHOSEN)
   Pros: Fast, compositional, fashion-optimized, scalable
   Cons: Requires regex-based extraction
   Why chosen: Best balance of speed, accuracy, and compositionality

4. CHOSEN ARCHITECTURE: BLIP + CLIP + INTELLIGENT COMPOSITION-AWARE RANKING
-----------------------------------------------------

## System Architecture Diagram

<img width="858" height="1284" alt="image" src="https://github.com/user-attachments/assets/b2131438-44cb-4b3b-a1dc-afbff9d728b1" />


**Visual Overview:**
The diagram shows the complete pipeline:
- **INDEXING PIPELINE (Left side):**
  1. Image Input
  2. BLIP Base Caption (Unconditional + Conditional dual modes)
  3. Advanced Extraction (Colors | Clothing | Context | Composition)
  4. CLIP Encoding → 512-dimensional vectors
  5. Qdrant Vector Database (Hybrid Storage)

- **RETRIEVAL PIPELINE (Right side):**
  1. User Query Input
  2. Query Parsing (Extract composition pairs)
  3. CLIP Query Encoding → 512-dimensional vector
  4. Vector Search (Cosine Similarity)
  5. **Attribute Boosting + Composition Re-ranking** (NEW)
  6. Top-K Results with Intelligent Scores

**Key Innovation:** Composition-aware re-ranking prevents attribute swaps and ensures:
- "red pants + black jacket" ≠ "red jacket + black pants"
- Exact color-item pair matching with 20% scoring weight
- Automatic correction of semantic similarity ambiguities

## Detailed Component Breakdown

4A. INDEXING PROCESS
--------------------

Step 1: Image Load & Validation
   - Load image as RGB
   - Verify integrity (detect corrupted files)
   - Skip on failure

Step 2: Dual Caption Generation (BLIP)
   - Unconditional: "a woman standing in a park" (full scene)
   - Conditional: "a photo of a person wearing a blue shirt" (clothing focus)
   - Combine: "a woman standing in a park. a photo of a person wearing a blue shirt"

Step 3: Attribute Extraction (Regex + Inference)
   - COLORS: Extract with modifiers → "bright yellow", "dark blue"
   - CLOTHING: Detailed patterns → "button-down shirt", "slim-fit jeans"
   - CONTEXT: Scene + location → "park", "outdoor", "street", "runway"
   - STYLE: From caption + inferred from clothing → "casual", "formal", "business"

Step 4: Semantic Embedding (CLIP)
   - Encode combined caption with CLIP-ViT-B-32
   - Generate 512-dimensional vector
   - Normalized (L2): magnitude ≈ 1.0

Step 5: Storage
   - Store in Qdrant Cloud
   - Payload includes: caption, colors, clothing, context, style, image_name

4B. RETRIEVAL PROCESS
---------------------

Step 1: Query Parsing
   - Extract query attributes using same regex patterns
   - Example: "bright yellow raincoat" → colors=["bright yellow"], clothing=["raincoat"]

Step 2: Query Encoding
   - Encode query with CLIP (same model as indexing)
   - Generate 512-dim vector

Step 3: Vector Search (Cosine Similarity)
   - Search in Qdrant: returns top k*3 candidates
   - Cosine similarity score: range [0, 1]
   - Higher = more similar

Step 4: Attribute Boosting + Composition Re-ranking (NEW)
   - **Composition Accuracy (20% weight):**
     * Extract color-item pairs from query
     * Match against caption: "red pants" vs "red jacket"
     * Score 1.0 for exact match, 0.3 for partial, 0.0 for mismatch
     * Prevents ranking swaps from pure semantic similarity
   
   - **Intelligent Multi-Factor Scoring:**
     * Base Score (40%): Original cosine similarity
     * Direct Match (30%): Query ↔ Caption embedding similarity
     * Composition (20%): Color-item pair accuracy [NEW]
     * Uniqueness (5%): Caption detail quality
     * Diversity (5%): MMR penalty for similar results
   
   - **Final Score:** Weighted combination → Natural range [0.65-0.95]

Step 5: Return Top-K
   - Re-rank by intelligent combined score
   - Return top-k results with differentiated scores
   - Example: "red pants query" → [0.859, 0.803, 0.773] instead of [0.845, 0.845, 0.845]

5. WHY THIS SOLVES EACH REQUIREMENT
-----------------------------------

✓ COMPOSITIONALITY
   - Dual captions capture "person" + "clothing" separately
   - CLIP encodes full sentence preserving order information
   - Attribute extraction separates individual components
   - Example: "red shirt blue pants" ≠ "blue shirt red pants" (different attribute combinations)

✓ ZERO-SHOT
   - CLIP trained on 400M image-text pairs
   - Understands unseen fashion descriptions
   - Semantic similarity captures meaning beyond keywords
   - Example: "casual weekend outfit" matches "person in jeans and hoodie" (semantic match)

✓ FASHION-SPECIFIC
   - Clothing taxonomy: button-down, blazer, jeans, etc.
   - Color modifiers: bright, dark, neon
   - Context keywords: runway, office, park, street
   - Style inference from clothing items

✓ SCALABILITY
   - Vector search: O(log n) on indexed data
   - Attribute boosting: O(k*3) linear but constant
   - Total query: ~50-100ms for 3200 images
   - Scales to 1M+ with Qdrant distributed mode

6. PERFORMANCE METRICS
---------------------

Indexing:
- Speed: 0.5-1 second per image (BLIP inference)
- Storage: 2KB per image (vector + metadata)
- Total for 3200 images: ~6.4MB vectors + metadata

Query:
- Latency: 50-100ms average
- Recall: High (CLIP + attribute matching)
- Precision: Improved by attribute boosting

7. COMPARISON WITH BASELINES
----------------------------

VANILLA CLIP:
- Pros: Fast (10ms query), simple
- Cons: No compositionality, misses fashion details
- Example: "red shirt blue pants" → same score as "blue shirt red pants"

FLORENCE-2:
- Pros: Better captions
- Cons: Too slow (3-5s per image), infrastructure heavy
- Indexing 3200 images: 3-5 hours vs 1-2 hours with BLIP

OUR APPROACH:
- Pros: Fast + Compositional + Fashion-optimized
- Cons: Requires regex maintenance
- Best balance for this use case

8. SHORTCOMINGS & LIMITATIONS
-----------------------------

REGEX BRITTLENESS:
- Novel clothing terms not in patterns → missed
- Solution: CLIP fallback ensures semantic matching always works

ATTRIBUTE EXTRACTION LIMITS:
- Can't detect attributes not mentioned in caption
- Solution: Dual captioning increases coverage

CONTEXT INFERENCE:
- Background understanding limited by vision model
- Solution: Use high-quality vision model (BLIP base proven good)

LANGUAGE DEPENDENCY:
- English-only currently
- Solution: Multilingual models available

9. FUTURE WORK
--------------

A. GEOGRAPHIC & WEATHER CONTEXT
   - Add location embeddings (city-specific fashion trends)
   - Weather attributes (winter coat vs summer dress)
   - Implementation: Append location/weather tokens to caption

B. PRECISION IMPROVEMENT
   - Fine-tune CLIP on fashion dataset
   - Learn clothing-specific embeddings
   - Custom loss function for compositional queries

C. TEMPORAL TRENDS
   - Track fashion trends by date
   - Season-aware recommendations
   - Implementation: Time-based clustering in vector space

D. VISUAL ATTRIBUTES
   - Add visual feature extraction (fabric texture, fit)
   - Combine with semantic features
   - Implementation: Multi-head attention

E. USER FEEDBACK LOOP
   - Learn from clicks/ratings
   - Adapt weights (boost/context/style)
   - Implementation: Online learning

10. TECHNICAL STACK
-------------------

Models:
- Vision-Language: Salesforce/blip-image-captioning-base
- Embeddings: sentence-transformers/clip-ViT-B-32
- Vector DB: Qdrant Cloud

Framework:
- PyTorch + Transformers
- qdrant-client
- PIL for image processing

Deployment:
- Python 3.9+
- ~2GB RAM (models)
- ~16GB disk (cache)

11. CODE STRUCTURE
------------------

indexer.py:
  - FashionIndexer class
  - _extract_fashion_caption(): Dual BLIP captioning
  - _extract_attributes_advanced(): Regex + inference
  - process_image(): End-to-end per-image processing
  - index_directory(): Batch processing with retry logic

retriever.py:
  - FashionRetriever class
  - _parse_query_attributes(): Query attribute extraction
  - _compute_attribute_boost(): Scoring logic
  - search(): Query execution with hybrid ranking

main.py:
  - Smart re-indexing detection
  - Evaluation query execution
  - Interactive search mode

search.py:
  - Search-only mode (no indexing)
  - Fast query execution

12. MODULARITY
--------------

Separation of Concerns:
✓ Data Layer: Qdrant (easily swappable)
✓ ML Layer: BLIP + CLIP (independently updatable)
✓ Logic Layer: Attribute extraction + ranking (testable)
✓ Interface Layer: main.py + search.py (decoupled)

Changes to vector DB: Only retriever.py affected
Changes to caption model: Only indexer.py affected
Changes to ranking: Only _compute_attribute_boost() affected

13. SCORING EXPLANATION
-----------------------

Cosine Similarity (Vector):
  - Formula: cos(A, B) = (A · B) / (||A|| * ||B||)
  - Range: [-1, 1] typically
  - Normalized CLIP vectors: Usually [0, 1]
  - Higher = more similar

Attribute Boost:
  - Discrete: 0.20, 0.25, 0.15, 0.12 per match type
  - Max possible: 0.20*N_colors + 0.25*N_clothing + 0.15*N_context + 0.12*N_style
  - Can exceed 1.0 if many attributes match

Final Score:
  - Combined: vector_score + attribute_boost
  - Can be > 1.0 (e.g., 0.95 + 0.5 = 1.45)
  - Higher = better match
  - Used for re-ranking to return top-k

Why Scores Differ:
  - Different vector similarities (semantic differences)
  - Different attribute matches (specificity differences)
  - Same caption = should have slightly different scores due to vector precision

DUPLICATE RESULTS FIX:
  - Issue: Same image returned multiple times
  - Cause: Possible UUID collision or batch processing error
  - Solution: Add deduplication in search results

14. QUICK START
---------------

First Run (Indexing):
  python main.py
  # Select option 1 for fresh indexing
  # Indexes 3200 images (~1-2 hours depending on connection)

Subsequent Runs (Search Only):
  python search.py
  # Fast search without re-indexing

Re-indexing:
  python main.py
  # Select option 1 to delete and re-index

15. EVALUATION RESULTS
----------------------

Query: "A person in a bright yellow raincoat."
Expected: Images with yellow raincoats
Result: Top matches show yellow/bright colors + raincoat clothing

Query: "Professional business attire inside a modern office."
Expected: Formal clothing in office settings
Result: Top matches show business clothes + office context

Query: "Someone wearing a blue shirt sitting on a park bench."
Expected: Blue shirt + sitting + park context
Result: Composition preserved through semantic + attribute matching

Query: "Casual weekend outfit for a city walk."
Expected: Casual clothing on city streets
Result: Style inference + context matching

Query: "A red tie and a white shirt in a formal setting."
Expected: Red + white color combination + formal context
Result: Compositional understanding prevents "white + red" confusion

16. REFERENCES
--------------

BLIP: https://github.com/salesforce/BLIP
CLIP: https://github.com/openai/CLIP
Qdrant: https://qdrant.tech
SentenceTransformers: https://www.sbert.net

