import time
import chromadb
import re
from datasets import load_dataset
import numpy as np

# Step 1: Initialize the Chroma client
client = chromadb.PersistentClient(path="./")

# Step 2: Create a collection in Chroma (equivalent to a table)
def create_collection(collection_name, distance_metric):
    try:
        collection = client.get_or_create_collection(name=collection_name, metadata={"hnsw:space": distance_metric})
        print(f"Collection '{collection_name}' created successfully.")
        return collection
    except Exception as e:
        print(f"Error creating collection: {e}")
        return None

# Step 3: Load and split the BookCorpus dataset into sentences
def load_and_split_bookcorpus():
    print("Loading BookCorpus dataset...")
    dataset = load_dataset("bookcorpus", split="train[:1%]", trust_remote_code=True)
    print("Dataset loaded!")

    sentence_splitter = re.compile(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s')
    sentences = []
    for entry in dataset["text"]:
        sentences.extend(sentence_splitter.split(entry))  # Split paragraphs into sentences
    
    return sentences

# Step 4: Insert the sentences into the Chroma collection
def insert_sentences_into_chroma(collection, sentences):
    try:
        times = []
        batch_size = 1
        for i in range(0, len(sentences), batch_size):
            batch_sentences = sentences[i:i+batch_size]
            batch_ids = [str(j) for j in range(i, i + len(batch_sentences))]
            start_time = time.time()
            collection.add(documents=batch_sentences, ids=batch_ids)
            end_time = time.time()
            insertion_time = end_time - start_time
            times.append(insertion_time)
            print(f"Inserted batch {i // batch_size + 1} with {len(batch_sentences)} sentences")
        print(f"Total of {len(sentences)} sentences successfully inserted into the collection!")
    except Exception as e:
        print(f"Error inserting sentences: {e}")
    return times

# Step 5: Compute and display statistics for the recorded times
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
    # Step 1: Initialize the Chroma client
    client = chromadb.Client()

    # Step 2: Create or connect to a collection
    collection_name1 = "bookcorpus_sentences1"
    collection1 = create_collection(collection_name1, "cosine")
    selected_sentence_ids = [9, 10, 11, 12, 13, 14, 15, 16, 17, 18]

    if collection1:

        # Step 3: Load and split the BookCorpus dataset into sentences
        sentences = load_and_split_bookcorpus()

        # Step 4: Insert the sentences into Chroma (adjust the number based on memory limits)
        times = insert_sentences_into_chroma(collection1, sentences[:10000])

        # Step 5: Compute and display statistics for the recorded times
        compute_statistics(times)

    else:
        print("Failed to create or connect to the collection.")
