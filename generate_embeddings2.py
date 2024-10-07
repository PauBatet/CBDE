import psycopg2
from transformers import AutoTokenizer, TFAutoModel
import numpy as np
import tensorflow as tf

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
        raise

def generate_embeddings():
    # Connectar bd
    conn = connect_to_db()
    cursor = conn.cursor()

    # Carrega model i tokenizer
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = TFAutoModel.from_pretrained(model_name)

    # Obtenir oracions
    cursor.execute("SELECT id, sentence FROM sentences")
    sentences = cursor.fetchall()

    for sentence_id, sentence in sentences:
        # Tokenitzar i generar embedding
        inputs = tokenizer(sentence, return_tensors='tf')  #TensorFlow
        outputs = model(**inputs)
        embedding = outputs.last_hidden_state.numpy().mean(axis=1)  # Obtenir embedding i calc mitjana

        # Guardar embedding en la bd
        cursor.execute(
            "UPDATE sentences SET embedding = %s WHERE id = %s",
            (embedding.tobytes(), sentence_id)
        )

        print(f"Processed sentence ID {sentence_id}")

    # Guardar i tancar
    conn.commit()
    cursor.close()
    conn.close()

if __name__ == "__main__":
    generate_embeddings()
