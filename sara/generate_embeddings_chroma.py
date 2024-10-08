import psycopg2
from sentence_transformers import SentenceTransformer
import chromadb

def connect_to_db():
    try:
        conn = psycopg2.connect(
            dbname='suppliers', 
            user='postgres',     
            password='123456',   
            host='localhost',    
            port='5432'          
        )
        return conn
    except Exception as e:
        print(f"Error al conectar a la base de datos: {e}")
        return None

def generate_embeddings(sentences, model):
    print(f"Generando embeddings para {len(sentences)} sentencias...")
    embeddings = model.encode(sentences, show_progress_bar=True)
    return embeddings

def store_embeddings_in_chroma(collection, sentences, embeddings, batch_size=5000):
    try:
        total_sentences = len(sentences)
        for i in range(0, total_sentences, batch_size):
            batch_sentences = sentences[i:i+batch_size]
            batch_embeddings = embeddings[i:i+batch_size]
            batch_ids = [f"sentence_{i+j}" for j in range(len(batch_sentences))]  # IDs unics
            
            collection.add(
                documents=batch_sentences,
                ids=batch_ids,
                embeddings=batch_embeddings.tolist()
            )
            print(f"Se almacenaron correctamente {len(batch_sentences)} sentencias en Chroma (batch {i//batch_size + 1}).")
    except Exception as e:
        print(f"Error almacenando embeddings en Chroma: {e}")


if __name__ == "__main__":
    conn = connect_to_db()
    if conn is None:
        print("No se pudo conectar a la base de datos. Saliendo.")
        exit()

    cursor = conn.cursor()
    cursor.execute("SELECT id, sentence FROM sentences") 
    rows = cursor.fetchall()

    if not rows:
        print("No se encontraron oraciones en la base de datos.")
        exit()

    sentences = [row[1] for row in rows]
    print(f"Se recuperaron {len(sentences)} oraciones de la base de datos.")

    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("Modelo de embeddings cargado correctamente.")

    embeddings = generate_embeddings(sentences, model)

    client = chromadb.Client()
    collection = client.create_collection(name="sentences_collection")
    print("Colección en Chroma creada exitosamente.")

    store_embeddings_in_chroma(collection, sentences, embeddings)

    conn.close()
    print("Conexión a la base de datos cerrada.")
