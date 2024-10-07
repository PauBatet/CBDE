import os
import glob

def borrar_chunks(save_dir):
    # glob per trobar arxius que començen per "chunk_"
    archivos_chunk = glob.glob(os.path.join(save_dir, "chunk_*.pkl"))
    
    for archivo in archivos_chunk:
        os.remove(archivo)
        print(f"Eliminado: {archivo}")

if __name__ == "__main__":
    save_dir = "../data/processed/embeddings_chunks/"  # ruta on es guarden
    borrar_chunks(save_dir)
