import psycopg2
from config import load_config
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
import time

BATCH_SIZE = 1000  # Number of sentences to process in each batch
TOTAL_SENTENCES = 10000  # Total number of sentences to process (first 10k)

def connect_to_db(config):
    """Connect to the PostgreSQL database."""
    try:
        conn = psycopg2.connect(**config)
        print("Connected to the PostgreSQL server.")
        return conn
    except Exception as error:
        print(f"Failed to connect to database: {error}")
        return None

def add_embedding_column(conn):
    """Add an embedding column to the bookcorpus_sentences table if it doesn't exist."""
    try:
        with conn.cursor() as cur:
            cur.execute("""
                ALTER TABLE bookcorpus_sentences 
                ADD COLUMN IF NOT EXISTS embedding double precision[];
            """)
            conn.commit()
            print("Added 'embedding' column to bookcorpus_sentences (if not existed).")
    except Exception as e:
        print(f"Failed to add 'embedding' column: {e}")
        conn.rollback()

def fetch_sentences_batch(conn, batch_size, offset):
    """Fetch a batch of sentences from the bookcorpus_sentences table."""
    try:
        with conn.cursor() as cur:
            cur.execute(f"""
                SELECT id, sentence FROM bookcorpus_sentences
                ORDER BY id
                LIMIT {batch_size} OFFSET {offset};
            """)
            rows = cur.fetchall()
            return rows
    except Exception as e:
        print(f"Failed to fetch sentences: {e}")
        return []

def generate_sentence_embeddings(sentences, model, tokenizer):
    """Generate embeddings for a list of sentences using a pre-trained model."""
    model.eval()
    with torch.no_grad():
        inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)  # Take the mean of the hidden states
        return embeddings.numpy()  # Convert to numpy array

def store_embedding(conn, sentence_id, embedding):
    """Update a single sentence in the database with its corresponding embedding."""
    try:
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE bookcorpus_sentences 
                SET embedding = %s 
                WHERE id = %s;
            """, (embedding.tolist(), sentence_id))
            conn.commit()
    except Exception as e:
        print(f"Failed to store embedding for sentence ID {sentence_id}: {e}")
        conn.rollback()

def compute_statistics(times, label):
    """Compute and display statistics for the recorded times."""
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

    add_embedding_column(conn)

    model_name = 'sentence-transformers/all-MiniLM-L6-v2'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    print(f"Processing up to {TOTAL_SENTENCES} sentences in batches of {BATCH_SIZE}...")
    offset = 0
    total_processed = 0

    # Lists to store time taken for generating embeddings and storing in the database
    embedding_times = []
    storage_times = []

    while total_processed < TOTAL_SENTENCES:
        # Fetch a batch of sentences
        sentences_data = fetch_sentences_batch(conn, BATCH_SIZE, offset)
        if not sentences_data:
            print("No more sentences to process.")
            break

        sentence_ids, sentences = zip(*sentences_data)

        # Measure time for generating embeddings
        print(f"Generating embeddings for batch starting at offset {offset}...")
        start_time = time.time()
        sentence_embeddings = generate_sentence_embeddings(sentences, model, tokenizer)
        embedding_time = time.time() - start_time
        embedding_times.append(embedding_time)

        # Store embeddings one by one and measure time
        for sentence_id, embedding in zip(sentence_ids, sentence_embeddings):
            start_time = time.time()
            store_embedding(conn, sentence_id, embedding)
            storage_time = time.time() - start_time
            storage_times.append(storage_time)

        # Update counters and offset
        total_processed += len(sentences_data)
        offset += len(sentences_data)

        print(f"Processed {total_processed} sentences so far.")

    compute_statistics(embedding_times, "Embedding Generation")
    compute_statistics(storage_times, "Database Storage (Single Inserts)")

    # Close the database connection
    conn.close()
    print("Batch processing completed successfully!")

if __name__ == "__main__":
    main()
