import os
import spacy
import pickle
import psycopg2
from config import load_config
from connect import connect  # Asegúrate de importar la función connect

# Cargar spaCy para dividir en oraciones
nlp = spacy.load("en_core_web_sm")

def dividir_en_oraciones(text):
    doc = nlp(text)
    return [sent.text for sent in doc.sents]

def insertar_oraciones_en_db(conn, chunk_id, oraciones):
    with conn.cursor() as cursor:
        for sentence in oraciones:
            try:
                cursor.execute(
                    "INSERT INTO sentences (chunk_id, sentence) VALUES (%s, %s)",
                    (chunk_id, sentence)
                )
            except Exception as e:
                print(f"Error al insertar la oración: {sentence} \nError: {e}")  # Imprime el error
        conn.commit()

def procesar_chunks_y_cargar_en_db():
    # Cargar la configuración
    config = load_config()
    # Conectar a la base de datos
    conn = connect(config)

    # Cargar y procesar cada chunk
    chunks_dir = "../data/processed/embeddings_chunks/" # Asegúrate de que esta ruta sea correcta
    chunk_id = 1

    for chunk_filename in os.listdir(chunks_dir):
        if chunk_filename.endswith(".pkl"):
            with open(os.path.join(chunks_dir, chunk_filename), 'rb') as f:
                chunk = pickle.load(f)
                print(f"Contenido de {chunk_filename}: {chunk}")  # Imprime el contenido del chunk
            
            # Dividir los textos del chunk en oraciones
            oraciones = []
            for texto in chunk['text']:  # Asegúrate de que `chunk` tenga esta estructura
                oraciones.extend(dividir_en_oraciones(texto))
            
            # Insertar las oraciones en PostgreSQL
            if oraciones:  # Verifica que hay oraciones para insertar
                print(f"Inserción de {len(oraciones)} oraciones del chunk {chunk_id}.")
                insertar_oraciones_en_db(conn, chunk_id, oraciones)
            else:
                print(f"No se encontraron oraciones para el chunk {chunk_id}.")
                
            chunk_id += 1

    # Cerrar la conexión
    conn.close()

if __name__ == '__main__':
    procesar_chunks_y_cargar_en_db()
