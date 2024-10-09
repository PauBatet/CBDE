# File: compute_similarities_all.py

import psycopg2
from config import load_config
import numpy as np
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
                cur.execute("""SELECT id, sentence, embedding FROM bookcorpus_sentences WHERE id = ANY(%s);""", (sentence_ids,))
            else:
                # Fetch embeddings for all sentences
                cur.execute("""SELECT id, sentence, embedding FROM bookcorpus_sentences;""")
            rows = cur.fetchall()
            return rows
    except Exception as e:
        print(f"Failed to fetch embeddings: {e}")
        return []

def compute_inner_product(a, b):
    """Compute inner product between two arrays."""
    return np.dot(a, b)

def compute_cosine_similarity(a, b):
    """Compute cosine similarity between two arrays."""
    return compute_inner_product(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def compute_similarities(selected_data, all_data):
    """
    Compute the top-2 similar sentences for each of the selected sentences
    using cosine similarity and inner product. Record time for each calculation.
    """
    selected_ids, selected_sentences, selected_embeddings = zip(*selected_data)
    all_ids, all_sentences, all_embeddings = zip(*all_data)

    # Convert embeddings to numpy arrays
    selected_embeddings = np.array([np.array(embedding) for embedding in selected_embeddings])
    all_embeddings = np.array([np.array(embedding) for embedding in all_embeddings])

    results = {}
    cosine_times = []
    inner_product_times = []

    # Iterate over each of the selected sentences
    for i, (selected_id, selected_embedding) in enumerate(zip(selected_ids, selected_embeddings)):
        # Compute cosine similarities
        start_time = time.time()
        cosine_similarities = np.array([compute_cosine_similarity(selected_embedding, emb) for emb in all_embeddings])
        cosine_time = time.time() - start_time
        cosine_times.append(cosine_time)

        # Compute inner products
        start_time = time.time()
        inner_products = np.array([compute_inner_product(selected_embedding, emb) for emb in all_embeddings])
        inner_product_time = time.time() - start_time
        inner_product_times.append(inner_product_time)

        # Get the indices of the top-2 most similar sentences (excluding self)
        cosine_indices = np.argsort(cosine_similarities)[-3:-1]  # Top 2 (excluding self)
        inner_product_indices = np.argsort(inner_products)[-3:-1]  # Top 2 (excluding self)

        results[selected_id] = {
            'sentence': selected_sentences[i],
            'cosine_similarity': [(all_ids[idx], all_sentences[idx]) for idx in cosine_indices],
            'inner_product': [(all_ids[idx], all_sentences[idx]) for idx in inner_product_indices],
        }

    return results, cosine_times, inner_product_times

def compute_statistics(times, label):
    if times:
        print(f"\n--- {label} Time Statistics ---")
        print(f"Minimum Time: {np.min(times):.6f} seconds")
        print(f"Maximum Time: {np.max(times):.6f} seconds")
        print(f"Average Time: {np.mean(times):.6f} seconds")
        print(f"Standard Deviation: {np.std(times):.6f} seconds")
    else:
        print(f"No {label} times recorded.")

def main():
    config = load_config()
    conn = connect_to_db(config)
    if not conn:
        return

    selected_sentence_ids = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]  # Replace with actual IDs

    print("Fetching selected sentences...")
    selected_data = fetch_embeddings(conn, selected_sentence_ids)

    print("Fetching all 10k sentences...")
    all_data = fetch_embeddings(conn)

    print("Computing similarities...")
    results, cosine_times, inner_product_times = compute_similarities(selected_data, all_data)

    for sentence_id, similar_sentences in results.items():
        print(f"\nSentence ID {sentence_id}:")
        print(f"  Sentence: {similar_sentences['sentence']}")

        print("  Top 2 Cosine Similarities:")
        for similar_id, similar_sentence in similar_sentences['cosine_similarity']:
            print(f"    - ID: {similar_id}, Sentence: {similar_sentence}")

        print("  Top 2 Inner Products:")
        for similar_id, similar_sentence in similar_sentences['inner_product']:
            print(f"    - ID: {similar_id}, Sentence: {similar_sentence}")

    compute_statistics(cosine_times, "Cosine Similarity Calculation")
    compute_statistics(inner_product_times, "Inner Product Calculation")

    # Close the database connection
    conn.close()
    print("Similarity computation completed successfully!")

if __name__ == "__main__":
    main()
