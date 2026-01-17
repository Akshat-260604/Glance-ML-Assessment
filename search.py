"""
Retrieval-only script - searches existing indexed data without re-indexing
Run this to query the Qdrant database directly
"""
import os
import warnings
from retriever import FashionRetriever

warnings.filterwarnings("ignore")

URL = "https://318891e2-d60f-448b-a61d-c64b87a9b6ab.europe-west3-0.gcp.cloud.qdrant.io:6333"
API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.eBMIhU7Yy_ShRfsz0YljJZgscA12Amb3iN34sQuLu5w"

def main():
    print("Connecting to Qdrant Fashion Index...")
    searcher = FashionRetriever(URL, API_KEY)

    print("\n--- Assignment Evaluation Queries ---")
    eval_queries = [
        "A person in a bright yellow raincoat.",
        "Professional business attire inside a modern office.",
        "Someone wearing a blue shirt sitting on a park bench.",
        "Casual weekend outfit for a city walk.",
        "A red tie and a white shirt in a formal setting."
    ]
    
    for q in eval_queries:
        print(f"\nSearching for: '{q}'")
        try:
            matches = searcher.search(q, k=2)
            if not matches:
                print("   No matches found.")
                continue
            for m in matches:
                payload = m.payload if hasattr(m, 'payload') else m.metadata
                score = m.score if hasattr(m, 'score') else "N/A"
                print(f"   Match: {payload['image_name']} | Score: {round(score, 3)}")
                print(f"   Caption: {payload.get('caption', 'N/A')[:80]}...")
                print(f"   Attributes: Colors={payload.get('colors', [])}, Clothing={payload.get('clothing', [])}, Context={payload.get('context', [])}")
        except Exception as e:
            print(f"   Error searching: {e}")

    print("\n--- Live Interactive Mode ---")
    print("(Type 'exit' to quit)")
    while True:
        try:
            user_q = input("\nEnter what you are looking for: ")
            if user_q.lower() == 'exit':
                break
            
            results = searcher.search(user_q, k=3)
            if not results:
                print("   No matches found.")
                continue
            
            for i, res in enumerate(results):
                payload = res.payload if hasattr(res, 'payload') else res.metadata
                score = res.score if hasattr(res, 'score') else "N/A"
                print(f"\n   [{i+1}] {payload['image_name']} (Score: {round(score, 3)})")
                print(f"       Caption: {payload.get('caption', 'N/A')[:100]}...")
                print(f"       Colors: {payload.get('colors', [])}")
                print(f"       Clothing: {payload.get('clothing', [])}")
                print(f"       Context: {payload.get('context', [])}")
                print(f"       Style: {payload.get('style', [])}")
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"   Error: {e}")

if __name__ == "__main__":
    main()
