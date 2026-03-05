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
                    id TEXT PRIMARY KEY,
                    name TEXT,
                    embedding BLOB
                )
            """)

    def insert_employee(self, emp_id: str, name: str, embedding: np.ndarray):
        embedding_blob = embedding.astype(np.float32).tobytes()

        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO employees (id, name, embedding) VALUES (?, ?, ?)",
                (emp_id, name, embedding_blob)
            )

    def get_all_embeddings(self):
        with self._connect() as conn:
            rows = conn.execute("SELECT id, name, embedding FROM employees").fetchall()

        result = []
        for emp_id, name, blob in rows:
            embedding = np.frombuffer(blob, dtype=np.float32)
            result.append((emp_id, name, embedding))

        return result