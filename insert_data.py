import psycopg2
import re
import time
import numpy as np  # For computing statistics
from datasets import load_dataset
from config import load_config
from connect import connect

# Step 1: Define a function to create the table (if it doesn't already exist)
def create_table(conn):
    create_table_query = """
    CREATE TABLE IF NOT EXISTS bookcorpus_sentences (
        id SERIAL PRIMARY KEY,
        sentence TEXT NOT NULL
    );
    """
    try:
        with conn.cursor() as cursor:
            cursor.execute(create_table_query)
            conn.commit()
            print("Table 'bookcorpus_sentences' created successfully (if not existed).")
    except Exception as e:
        print(f"Error creating table: {e}")


# Step 2: Define a function to clear the table
def clear_table(conn):
    try:
        with conn.cursor() as cursor:
            cursor.execute("TRUNCATE TABLE bookcorpus_sentences;")
            conn.commit()
            print("Table 'bookcorpus_sentences' cleared successfully.")
    except Exception as e:
        print(f"Error clearing table: {e}")


# Step 3: Define a function to load and split the BookCorpus dataset into sentences
def load_and_split_bookcorpus():
    # Load a small subset of the BookCorpus dataset
    print("Loading BookCorpus dataset...")
    dataset = load_dataset("bookcorpus", split="train[:1%]", trust_remote_code=True)  # Load only 1% to avoid large memory usage
    print("Dataset loaded!")

    # Extract sentences and split into individual sentences
    sentence_splitter = re.compile(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s')
    sentences = []
    for entry in dataset["text"]:
        sentences.extend(sentence_splitter.split(entry))  # Split paragraphs into sentences
    
    return sentences


# Step 4: Define a function to insert the sentences into the PostgreSQL database and record the insertion times
def insert_sentences_with_timing(conn, sentences):
    times = []  # To store individual insertion times

    try:
        with conn.cursor() as cursor:
            insert_query = "INSERT INTO bookcorpus_sentences (sentence) VALUES (%s)"
            
            # Insert each sentence and record the time taken for each insertion
            for sentence in sentences:
                start_time = time.time()
                cursor.execute(insert_query, (sentence,))
                conn.commit()
                end_time = time.time()
                
                # Calculate time taken and store it
                insertion_time = end_time - start_time
                times.append(insertion_time)

        print(f"{len(sentences)} sentences successfully inserted into the database!")
    except Exception as e:
        print(f"Error inserting sentences: {e}")

    return times


# Step 5: Compute and display statistics from the recorded times
def compute_statistics(times):
    if times:
        print("\n--- Insertion Time Statistics ---")
        print(f"Minimum Time: {np.min(times):.6f} seconds")
        print(f"Maximum Time: {np.max(times):.6f} seconds")
        print(f"Average Time: {np.mean(times):.6f} seconds")
        print(f"Standard Deviation: {np.std(times):.6f} seconds")
    else:
        print("No times recorded.")


# Step 6: Main function to execute the steps
if __name__ == "__main__":
    # Load the database configuration
    config = load_config()

    # Establish a connection to the PostgreSQL database
    conn = connect(config)

    if conn is not None:
        # Create the table
        create_table(conn)

        # Clear the table
        clear_table(conn)

        # Load and split the BookCorpus dataset into sentences
        sentences = load_and_split_bookcorpus()

        # Insert the first 10,000 sentences into the database (or fewer if there aren't 10,000) and record times
        times = insert_sentences_with_timing(conn, sentences[:10000])  # Adjust as needed based on memory limits or dataset size

        # Compute and display statistics
        compute_statistics(times)

        # Close the connection
        conn.close()
    else:
        print("Failed to connect to the database.")
