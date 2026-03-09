import numpy as np
from src.database.db_manager import DatabaseManager
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "data/embeddings.db")

db = DatabaseManager(DB_PATH)

NUM_EMPLOYEES = 300000
EMBEDDING_SIZE = 512
BATCH_SIZE = 5000


def generate_embedding():
    vec = np.random.randn(EMBEDDING_SIZE)
    vec = vec / np.linalg.norm(vec)
    return vec


def main():

    names_batch = []
    embeddings_batch = []

    for i in range(NUM_EMPLOYEES):

        name = f"Synthetic_{i}"
        embedding = generate_embedding()

        names_batch.append(name)
        embeddings_batch.append(embedding)

        if len(names_batch) == BATCH_SIZE:

            db.insert_many_embeddings(names_batch, embeddings_batch)

            print(f"{i+1} embeddings inserted")

            names_batch = []
            embeddings_batch = []

    # insert remaining
    if names_batch:
        db.insert_many_embeddings(names_batch, embeddings_batch)

    print("Done generating synthetic embeddings.")


if __name__ == "__main__":
    main()