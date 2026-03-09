import sqlite3
import numpy as np


class DatabaseManager:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._create_table()

    def _connect(self):
        return sqlite3.connect(self.db_path)

    def _create_table(self):
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS employees (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT,
                    embedding BLOB
                )
            """)

    def insert_employee(self, name: str, embedding: np.ndarray):
        embedding_blob = embedding.astype(np.float32).tobytes()

        with self._connect() as conn:
            conn.execute(
                "INSERT INTO employees (name, embedding) VALUES (?, ?)",
                (name, embedding_blob)
            )

    def insert_many_embeddings(self, names, embeddings):
        with self._connect() as conn:
            data = []

            for name, embedding in zip(names, embeddings):
                blob = embedding.astype(np.float32).tobytes()
                data.append((name, blob))

            conn.executemany(
                "INSERT INTO employees (name, embedding) VALUES (?, ?)",
                data
            )

    def get_all_embeddings(self):
        with self._connect() as conn:
            rows = conn.execute("SELECT name, embedding FROM employees").fetchall()

        result = []

        for name, blob in rows:
            embedding = np.frombuffer(blob, dtype=np.float32)
            result.append({
                "name": name,
                "embedding": embedding
            })

        return result