import chromadb
import re
from datasets import load_dataset

# Step 1: Initialize the Chroma client
client = chromadb.Client()

# Step 2: Create a collection in Chroma (equivalent to a table)
def create_collection(collection_name):
    try:
        collection = client.create_collection(name=collection_name)
        print(f"Collection '{collection_name}' created successfully.")
        return collection
    except Exception as e:
        print(f"Error creating collection: {e}")
        return None

# Step 3: Clear the collection (if needed)
def clear_collection(collection):
    try:
        collection.delete()
        print("Collection cleared successfully.")
    except Exception as e:
        print(f"Error clearing collection: {e}")

# Step 4: Load and split the BookCorpus dataset into sentences
def load_and_split_bookcorpus():
    print("Loading BookCorpus dataset...")
    dataset = load_dataset("bookcorpus", split="train[:1%]", trust_remote_code=True)
    print("Dataset loaded!")

    sentence_splitter = re.compile(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s')
    sentences = []
    for entry in dataset["text"]:
        sentences.extend(sentence_splitter.split(entry))  # Split paragraphs into sentences
    
    return sentences

# Step 5: Insert the sentences into the Chroma collection
def insert_sentences_into_chroma(collection, sentences):
    try:
        batch_size = 500  # Define the batch size
        for i in range(0, len(sentences), batch_size):
            batch_sentences = sentences[i:i+batch_size]  # Slice sentences into batches of 500
            batch_ids = [str(j) for j in range(i, i + len(batch_sentences))]  # Generate IDs for the current batch
            collection.add(documents=batch_sentences, ids=batch_ids)
            print(f"Inserted batch {i // batch_size + 1} with {len(batch_sentences)} sentences")
        print(f"Total of {len(sentences)} sentences successfully inserted into the collection!")
    except Exception as e:
        print(f"Error inserting sentences: {e}")

# Step 6: Query the collection to find the top-2 similar sentences for selected sentences
def query_top_similar_for_selected(collection, sentences, selected_ids, n_results=2):
    try:
        results = {}
        for selected_id in selected_ids:
            query_sentence = sentences[selected_id]  # Get the sentence corresponding to the selected index
            query_result = collection.query(
                query_texts=[query_sentence],
                n_results=n_results + 1  # Include self-match to exclude it later
            )

            # Extract results and exclude the self-match (the first result)
            similar_sentences = query_result['documents'][0][1:n_results+1]  # Skip self-match at index 0
            results[query_sentence] = similar_sentences

        return results
    except Exception as e:
        print(f"Error querying similar sentences: {e}")
        return None

# Step 7: Main function to execute the steps
if __name__ == "__main__":
    # Step 1: Initialize the Chroma client
    client = chromadb.Client()

    # Step 2: Create or connect to a collection
    collection_name = "bookcorpus_sentences"
    collection = create_collection(collection_name)
    selected_sentence_ids = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]

    if collection:
        # Step 3: Clear the collection (if necessary)
        clear_collection(collection)

        # Step 4: Load and split the BookCorpus dataset into sentences
        sentences = load_and_split_bookcorpus()

        # Step 5: Insert the sentences into Chroma (adjust the number based on memory limits)
        insert_sentences_into_chroma(collection, sentences[:10000])

        # Step 6: Query for the top-2 similar sentences for each selected sentence
        results = query_top_similar_for_selected(collection, sentences, selected_sentence_ids, n_results=2)
        print("Top 2 similar sentences for each selected sentence:")
        for query_sentence, similar_sentences in results.items():
            print(f"Query Sentence: {query_sentence}")
            print(f"Similar Sentences: {similar_sentences}")
            print("-" * 80)
            
    else:
        print("Failed to create or connect to the collection.")
