import numpy as np
import psycopg2
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import euclidean

def connect_to_db():
    conn = psycopg2.connect(
        dbname='suppliers',
        user='postgres',
        password='123456',
        host='localhost',
        port='5432'
    )
    return conn

# Recuperar oracions i embeddings
def get_sentences_and_embeddings(limit=10):
    conn = connect_to_db()
    cursor = conn.cursor()
    cursor.execute("SELECT id, sentence, embedding FROM sentences LIMIT %s", (limit,))
    data = cursor.fetchall()
    conn.close()
    
    sentences = []
    embeddings = []
    for row in data:
        sentence_id, sentence, embedding_bytes = row
        embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
        sentences.append((sentence_id, sentence))
        embeddings.append(embedding)
    
    return sentences, np.array(embeddings)

# Calcular las top-2 més similars
def compute_top2_similar(sentences, embeddings, metric):
    n = len(sentences)
    top_similarities = []
    
    for i in range(n):
        distances = []
        for j in range(n):
            if i != j:
                if metric == 'cosine':
                    dist = cosine_similarity([embeddings[i]], [embeddings[j]])[0][0]
                elif metric == 'euclidean':
                    dist = euclidean(embeddings[i], embeddings[j])
                distances.append((sentences[j][1], dist))
        
        if metric == 'cosine':
            # Majors més similitut
            top2 = sorted(distances, key=lambda x: x[1], reverse=True)[:2]
        elif metric == 'euclidean':
            # Menors més similitut
            top2 = sorted(distances, key=lambda x: x[1])[:2]
        
        top_similarities.append((sentences[i][1], top2))
    
    return top_similarities

if __name__ == "__main__":
    # Recuperar 
    sentences, embeddings = get_sentences_and_embeddings(10)
    
    # Top-2  similars cosine similarity
    top2_cosine = compute_top2_similar(sentences, embeddings, 'cosine')
    print("Top-2 similares usando Cosine Similarity:")
    for sentence, similar_sentences in top2_cosine:
        print(f"Oración: {sentence}")
        for sim_sentence, score in similar_sentences:
            print(f" - Similar: {sim_sentence}, Score: {score}")
    
    # Top-2 similars euclidean distance
    top2_euclidean = compute_top2_similar(sentences, embeddings, 'euclidean')
    print("\nTop-2 similares usando Euclidean Distance:")
    for sentence, similar_sentences in top2_euclidean:
        print(f"Oración: {sentence}")
        for sim_sentence, score in similar_sentences:
            print(f" - Similar: {sim_sentence}, Score: {score}")
