# File: compute_similarities_all.py

import psycopg2
from config import load_config
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from scipy.spatial.distance import cityblock  # Manhattan distance
import time

def connect_to_db(config):
    """Connect to the PostgreSQL database."""
    try:
        conn = psycopg2.connect(**config)
        print("Connected to the PostgreSQL server.")
        return conn
    except Exception as error:
        print(f"Failed to connect to database: {error}")
        return None

def fetch_embeddings(conn, sentence_ids=None):
    """
    Fetch embeddings from the database.
    If sentence_ids is provided, fetch embeddings only for those IDs.
    """
    try:
        with conn.cursor() as cur:
            if sentence_ids:
                # Fetch embeddings only for selected sentences
                cur.execute("""
                    SELECT id, sentence, embedding FROM bookcorpus_sentences
                    WHERE id = ANY(%s);
                """, (sentence_ids,))
            else:
                # Fetch embeddings for all sentences
                cur.execute("""
                    SELECT id, sentence, embedding FROM bookcorpus_sentences;
                """)
            rows = cur.fetchall()
            return rows
    except Exception as e:
        print(f"Failed to fetch embeddings: {e}")
        return []

def compute_similarities(selected_data, all_data):
    """
    Compute the top-2 similar sentences for each of the selected sentences using
    cosine similarity, Euclidean distance, and Manhattan distance.
    """
    selected_ids, selected_sentences, selected_embeddings = zip(*selected_data)
    all_ids, all_sentences, all_embeddings = zip(*all_data)

    # Convert embeddings to numpy arrays
    selected_embeddings = np.array([np.array(embedding) for embedding in selected_embeddings])
    all_embeddings = np.array([np.array(embedding) for embedding in all_embeddings])

    results = {}

    # Iterate over each of the 10 selected sentences
    for i, (selected_id, selected_embedding) in enumerate(zip(selected_ids, selected_embeddings)):
        # Compute similarities/distances against all 10k sentences
        euclidean_dist = euclidean_distances([selected_embedding], all_embeddings)[0]  # Use negative for highest similarity
        manhattan_dist = np.array([cityblock(selected_embedding, emb) for emb in all_embeddings])  # Use negative for highest similarity

        # Get the indices of the top-2 most similar sentences (excluding self)
        euclidean_indices = np.argsort(euclidean_dist)[1:3]
        manhattan_indices = np.argsort(manhattan_dist)[1:3]

        results[selected_id] = {
            'sentence': selected_sentences[i],
            'euclidean': [(all_ids[idx], all_sentences[idx]) for idx in euclidean_indices],
            'manhattan': [(all_ids[idx], all_sentences[idx]) for idx in manhattan_indices],
        }

    return results

def main():
    # Step 1: Load configuration and connect to the database
    config = load_config()
    conn = connect_to_db(config)
    if not conn:
        return

    # Step 2: Select 10 sentence IDs (replace with actual IDs from your table)
    selected_sentence_ids = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]  # Replace with actual IDs of the 10 sentences to compare

    # Step 3: Fetch embeddings for the selected sentences and all sentences
    print("Fetching selected sentences...")
    selected_data = fetch_embeddings(conn, selected_sentence_ids)

    print("Fetching all 10k sentences...")
    all_data = fetch_embeddings(conn)

    # Step 4: Compute similarities and measure time
    print("Computing similarities...")
    start_time = time.time()
    results = compute_similarities(selected_data, all_data)
    end_time = time.time()

    print(f"Time taken to compute similarities: {end_time - start_time:.2f} seconds")

    # Step 5: Display results
    for sentence_id, similar_sentences in results.items():
        print(f"\nSentence ID {sentence_id}:")
        print(f"  Sentence: {similar_sentences['sentence']}")

        print("  Top 2 Euclidean Distances:")
        for similar_id, similar_sentence in similar_sentences['euclidean']:
            print(f"    - ID: {similar_id}, Sentence: {similar_sentence}")

        print("  Top 2 Manhattan Distances:")
        for similar_id, similar_sentence in similar_sentences['manhattan']:
            print(f"    - ID: {similar_id}, Sentence: {similar_sentence}")

    # Close the database connection
    conn.close()
    print("Similarity computation completed successfully!")

if __name__ == "__main__":
    main()
