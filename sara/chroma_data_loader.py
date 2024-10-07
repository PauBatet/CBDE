import chromadb
import os
import pickle

client = chromadb.Client()

collection = client.create_collection(name="my_collection")

def cargar_chunks_en_chroma(chunks_dir):
    print(f"Buscando chunks en: {chunks_dir}")
    
    # Iterar sobre tots 
    for filename in os.listdir(chunks_dir):
        if filename.endswith(".pkl"):
            print(f"Encontrado chunk: {filename}")
            chunk_path = os.path.join(chunks_dir, filename)
            
            # Cargar chunk
            with open(chunk_path, 'rb') as f:
                chunk = pickle.load(f)
            
            texts = chunk['text'] 

            ids = [f"{filename[:-4]}_id_{i}" for i in range(len(texts))]  # Genera IDs únicos basados en el nombre del chunk

            # Afegir a Chroma
            collection.add(
                documents=texts,
                ids=ids  
            )
            print(f"Chunk {filename} cargado exitosamente.")

if __name__ == "__main__":
    # Ruta chuncks
    chunks_dir = "C:\\Users\\saram\\suppliers\\data\\processed\\embeddings_chunks" 

    cargar_chunks_en_chroma(chunks_dir)
