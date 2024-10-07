from datasets import load_dataset
from datasets import load_from_disk
from transformers import TFAutoModel, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import pickle
import tensorflow as tf
import numpy as np


def descargar_bookcorpus(save_path, limit=None):
    # Agrega trust_remote_code=True y un límite si se especifica
    dataset = load_dataset("bookcorpus", trust_remote_code=True)

    # Si se proporciona un límite, recorta el dataset
    if limit is not None:
        dataset['train'] = dataset['train'].select(range(limit))

    os.makedirs(save_path, exist_ok=True)
    
    # Guarda solo la partición 'train'
    dataset['train'].save_to_disk(save_path)
    print(f"Dataset guardado en {save_path}")


def dividir_chunks(dataset_path, chunk_size, save_dir):
    dataset = load_from_disk(dataset_path)
    
    # Asegúrate de que estás accediendo a los datos correctamente
    total_samples = len(dataset)  # Ya no necesitas ['train']
    total_chunks = total_samples // chunk_size + (total_samples % chunk_size > 0)
    
    os.makedirs(save_dir, exist_ok=True)
    for i in range(total_chunks):
        chunk = dataset[i * chunk_size:(i + 1) * chunk_size]  # Slicing directo del dataset
        with open(os.path.join(save_dir, f"chunk_{i}.pkl"), 'wb') as f:
            pickle.dump(chunk, f)
        print(f"Guardado chunk_{i}.pkl")
    print(f"Total de chunks creados: {total_chunks}")



def cargar_modelo(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = TFAutoModel.from_pretrained(model_name)
    print(f"Modelo {model_name} y tokenizer cargados correctamente.")
    return tokenizer, model

def get_embeddings(texts, tokenizer, model):
    # Tokenizar el texto
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="tf")
    
    # Obtener las salidas del modelo
    outputs = model(**inputs)
    
    # Promediar las representaciones de los tokens para obtener una representación de la oración
    embeddings = tf.reduce_mean(outputs.last_hidden_state, axis=1)
    return embeddings.numpy()

def procesar_chunks(chunks_dir, save_dir, tokenizer, model):
    os.makedirs(save_dir, exist_ok=True)
    for filename in os.listdir(chunks_dir):
        if filename.startswith("chunk_") and filename.endswith(".pkl"):
            with open(os.path.join(chunks_dir, filename), 'rb') as f:
                chunk = pickle.load(f)
            texts = chunk['text']  # Asegúrate de que el campo de texto se llama 'text'
            embeddings = get_embeddings(texts, tokenizer, model)
            with open(os.path.join(save_dir, f"embeddings_{filename[:-4]}.npy"), 'wb') as f:
                np.save(f, embeddings)
            print(f"Guardado embeddings_{filename[:-4]}.npy")

def get_embeddings(texts, tokenizer, model):
    # Tokenizar el texto
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="tf")
    
    # Obtener las salidas del modelo
    outputs = model(**inputs)
    
    # Promediar las representaciones de los tokens para obtener una representación de la oración
    embeddings = tf.reduce_mean(outputs.last_hidden_state, axis=1)
    return embeddings.numpy()

def cargar_embeddings(embeddings_dir):
    embeddings = []
    documentos = []
    for filename in os.listdir(embeddings_dir):
        if filename.startswith("embeddings_chunk_") and filename.endswith(".npy"):
            emb = np.load(os.path.join(embeddings_dir, filename))
            embeddings.append(emb)
            documentos.extend([filename] * emb.shape[0])  # Puedes ajustar esto según tu necesidad
    embeddings = np.vstack(embeddings)
    return embeddings, documentos

def buscar_similitud(query, tokenizer, model, embeddings, documentos, top_k=5):
    query_embedding = get_embeddings([query], tokenizer, model)
    similarities = cosine_similarity(query_embedding, embeddings)[0]
    top_k_indices = similarities.argsort()[-top_k:][::-1]
    top_k_similitudes = similarities[top_k_indices]
    top_k_documentos = [documentos[i] for i in top_k_indices]
    return top_k_documentos, top_k_similitudes

if __name__ == "__main__":
    save_path = "../data/raw/bookcorpus/"
    limit = 10000  # Cambia esto según lo que necesites
    descargar_bookcorpus(save_path, limit)

    dataset_path = "../data/raw/bookcorpus/"
    chunk_size = 1000  # Ajusta según tu memoria
    save_dir = "../data/processed/embeddings_chunks/"
    dividir_chunks(dataset_path, chunk_size, save_dir)

    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    tokenizer, model = cargar_modelo(model_name)

    chunks_dir = "../data/processed/embeddings_chunks/"
    embeddings_save_dir = "../data/processed/embeddings_chunks/"
    procesar_chunks(chunks_dir, embeddings_save_dir, tokenizer, model)

    query = "Ejemplo de consulta de búsqueda"
    embeddings_dir = "../data/processed/embeddings_chunks/"
    
    print("Cargando embeddings...")
    embeddings, documentos = cargar_embeddings(embeddings_dir)
    print(f"Total de embeddings cargados: {embeddings.shape[0]}")
    
    print("Buscando similitudes...")
    top_docs, top_sims = buscar_similitud(query, tokenizer, model, embeddings, documentos, top_k=5)
    
    print("Top 5 documentos más similares:")
    for doc, sim in zip(top_docs, top_sims):
        print(f"{doc} con similitud {sim:.4f}")