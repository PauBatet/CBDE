import psycopg2
from datasets import load_dataset
import re
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


# Step 2: Define a function to load and split the BookCorpus dataset into sentences
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


# Step 3: Define a function to insert the sentences into the PostgreSQL database
def insert_sentences(conn, sentences):
    try:
        with conn.cursor() as cursor:
            # Insert each sentence into the table
            insert_query = "INSERT INTO bookcorpus_sentences (sentence) VALUES (%s)"
            cursor.executemany(insert_query, [(sentence,) for sentence in sentences])
            conn.commit()
            print(f"{len(sentences)} sentences successfully inserted into the database!")
    except Exception as e:
        print(f"Error inserting sentences: {e}")


# Step 4: Main function to execute the steps
if __name__ == "__main__":
    # Load the database configuration
    config = load_config()

    # Establish a connection to the PostgreSQL database
    conn = connect(config)

    if conn is not None:
        # Create the table
        create_table(conn)

        # Load and split the BookCorpus dataset into sentences
        sentences = load_and_split_bookcorpus()

        # Insert the first 10,000 sentences into the database (or fewer if there aren't 10,000)
        insert_sentences(conn, sentences[:10000])  # Adjust as needed based on memory limits or dataset size

        # Close the connection
        conn.close()
    else:
        print("Failed to connect to the database.")
