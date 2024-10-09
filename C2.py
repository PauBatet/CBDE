import chromadb
import re
import time
import numpy as np
from datasets import load_dataset

# Step 1: Initialize the Chroma client
client = chromadb.PersistentClient(path="./")

# Step 2: Create a collection in Chroma (equivalent to a table)
def create_collection(collection_name, distance_metric):
    """
    Create or get a Chroma collection with the specified distance metric.

    Args:
        collection_name (str): The name of the collection.
        distance_metric (str): The distance metric to be used ('cosine' or 'ip').

    Returns:
        collection: The created or retrieved collection.
    """
    try:
        collection = client.get_or_create_collection(name=collection_name, metadata={"hnsw:space": distance_metric})
        print(f"Collection '{collection_name}' created successfully with distance metric '{distance_metric}'.")
        return collection
    except Exception as e:
        print(f"Error creating collection: {e}")
        return None

# Step 3: Load and split the BookCorpus dataset into sentences
def load_and_split_bookcorpus():
    """
    Load the BookCorpus dataset and split it into individual sentences.

    Returns:
        List[str]: A list of sentences extracted from the BookCorpus dataset.
    """
    print("Loading BookCorpus dataset...")
    dataset = load_dataset("bookcorpus", split="train[:1%]", trust_remote_code=True)
    print("Dataset loaded!")

    # Define a regex pattern to split paragraphs into sentences
    sentence_splitter = re.compile(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s')
    sentences = []
    for entry in dataset["text"]:
        sentences.extend(sentence_splitter.split(entry))  # Split paragraphs into sentences
    
    return sentences

# Step 4: Insert the sentences into the Chroma collection
def insert_sentences_into_chroma(collection, sentences):
    """
    Insert sentences into the specified Chroma collection in batches.

    Args:
        collection: The Chroma collection to insert sentences into.
        sentences (List[str]): The sentences to be inserted into the collection.
    """
    try:
        batch_size = 1000  # Define the batch size
        for i in range(0, len(sentences), batch_size):
            batch_sentences = sentences[i:i+batch_size]  # Slice sentences into batches of 1000
            batch_ids = [str(j) for j in range(i, i + len(batch_sentences))]  # Generate IDs for the current batch
            collection.add(documents=batch_sentences, ids=batch_ids)
            print(f"Inserted batch {i // batch_size + 1} with {len(batch_sentences)} sentences")
        print(f"Total of {len(sentences)} sentences successfully inserted into the collection!")
    except Exception as e:
        print(f"Error inserting sentences: {e}")

# Step 5: Query the collection to find the top-2 similar sentences for selected sentences
def query_top_similar_for_selected(collection, sentences, selected_ids, n_results=2):
    """
    Query the collection for the top-2 similar sentences for each selected sentence.

    Args:
        collection: The Chroma collection to be queried.
        sentences (List[str]): The list of sentences from which to query similar ones.
        selected_ids (List[int]): The indices of the sentences to be queried.
        n_results (int): The number of similar results to retrieve (excluding self-match).

    Returns:
        dict: A dictionary of query sentences and their corresponding top similar sentences.
        List[float]: A list of query times recorded for each query.
    """
    try:
        results = {}
        query_times = []  # To store the time taken for each query

        for selected_id in selected_ids:
            query_sentence = sentences[selected_id]  # Get the sentence corresponding to the selected index

            # Measure time taken for the query
            start_time = time.time()
            query_result = collection.query(
                query_texts=[query_sentence],
                n_results=n_results + 1  # Include self-match to exclude it later
            )
            query_time = time.time() - start_time
            query_times.append(query_time)

            # Extract results and exclude the self-match (the first result)
            similar_sentences = query_result['documents'][0][1:n_results+1]  # Skip self-match at index 0
            results[query_sentence] = similar_sentences

        return results, query_times
    except Exception as e:
        print(f"Error querying similar sentences: {e}")
        return None, []

# Step 6: Compute and display statistics for the recorded times
def compute_statistics(times, label):
    """
    Compute and display statistics for the recorded query times.

    Args:
        times (List[float]): A list of times recorded for each query.
        label (str): A label to identify the set of times (e.g., "Cosine Distance Query").
    """
    if times:
        print(f"\n--- {label} Time Statistics ---")
        print(f"Minimum Time: {np.min(times):.6f} seconds")
        print(f"Maximum Time: {np.max(times):.6f} seconds")
        print(f"Average Time: {np.mean(times):.6f} seconds")
        print(f"Standard Deviation: {np.std(times):.6f} seconds")
    else:
        print(f"No {label} times recorded.")

# Step 7: Main function to execute the steps
if __name__ == "__main__":
    # Step 1: Initialize the Chroma client
    client = chromadb.Client()

    # Step 2: Create or connect to collections with different similarity metrics
    cosine_collection_name = "bookcorpus_sentences_cosine"   # Collection for cosine similarity
    ip_collection_name = "bookcorpus_sentences_ip"           # Collection for inner product similarity

    cosine_collection = create_collection(cosine_collection_name, "cosine")
    ip_collection = create_collection(ip_collection_name, "ip")

    # Define sentence indices to be used for querying
    selected_sentence_ids = [9, 10, 11, 12, 13, 14, 15, 16, 17, 18]

    # Step 3: Load and split the BookCorpus dataset into sentences
    sentences = load_and_split_bookcorpus()

    # Initialize lists to store query times for both collections
    query_times_cosine = []
    query_times_ip = []

    if cosine_collection:
        # Step 4: Insert sentences into the cosine similarity collection
        insert_sentences_into_chroma(cosine_collection, sentences[:10000])

        # Step 5: Query for the top-2 similar sentences for each selected sentence and record query times
        results, query_times = query_top_similar_for_selected(cosine_collection, sentences, selected_sentence_ids, n_results=2)
        query_times_cosine.extend(query_times)

        print("Top 2 similar sentences for each selected sentence (Cosine Similarity):")
        for query_sentence, similar_sentences in results.items():
            print(f"Query Sentence: {query_sentence}")
            print(f"Similar Sentences: {similar_sentences}")
            print("-" * 80)

    if ip_collection:
        # Step 4: Insert sentences into the inner product similarity collection
        insert_sentences_into_chroma(ip_collection, sentences[:10000])

        # Step 5: Query for the top-2 similar sentences for each selected sentence and record query times
        results, query_times = query_top_similar_for_selected(ip_collection, sentences, selected_sentence_ids, n_results=2)
        query_times_ip.extend(query_times)

        print("Top 2 similar sentences for each selected sentence (Inner Product Similarity):")
        for query_sentence, similar_sentences in results.items():
            print(f"Query Sentence: {query_sentence}")
            print(f"Similar Sentences: {similar_sentences}")
            print("-" * 80)

    # Step 6: Display statistics for both collections
    compute_statistics(query_times_cosine, "Cosine Similarity Query")
    compute_statistics(query_times_ip, "Inner Product Similarity Query")
