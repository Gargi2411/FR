import numpy as np
from src.database.db_manager import DatabaseManager
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "data/embeddings.db")

db = DatabaseManager(DB_PATH)

NUM_EMPLOYEES = 300000
EMBEDDING_SIZE = 512

def generate_embedding():
    vec = np.random.randn(EMBEDDING_SIZE)
    vec = vec / np.linalg.norm(vec)
    return vec

def main():
    for i in range(NUM_EMPLOYEES):
        employee_id = f"emp_{i}"
        name = f"Synthetic_{i}"

        embedding = generate_embedding()

        db.insert_employee(employee_id, name, embedding)

        if i % 10000 == 0:
            print(f"{i} embeddings inserted")

    print("Done generating synthetic embeddings.")

if __name__ == "__main__":
    main()