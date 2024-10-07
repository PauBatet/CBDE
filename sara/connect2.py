import psycopg2
from config import load_config

def connect(config):
    """ Connect to the PostgreSQL database server """
    try:
        # connecting to the PostgreSQL server
        conn = psycopg2.connect(**config)
        print('Connected to the PostgreSQL server.')
        return conn
    except (psycopg2.DatabaseError, Exception) as error:
        print(error)

def crear_tabla_sentences(conn):
    """ Crea la tabla sentences si no existe """
    create_table_query = """
    CREATE TABLE IF NOT EXISTS sentences (
        id SERIAL PRIMARY KEY,
        chunk_id INT,
        sentence TEXT
    );
    """
    with conn.cursor() as cursor:
        cursor.execute(create_table_query)
        conn.commit()

if __name__ == '__main__':
    # Cargar la configuración
    config = load_config()
    
    # Conectar a la base de datos
    conn = connect(config)
    
    if conn:  # Verifica si la conexión fue exitosa
        crear_tabla_sentences(conn)  # Crear la tabla si no existe
        conn.close()  # Cerrar la conexión
