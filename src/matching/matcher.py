import numpy as np


class Matcher:
    def __init__(self, threshold=0.45):
        self.threshold = threshold

    def cosine_similarity(self, a, b):
        a = np.array(a)
        b = np.array(b)

        if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
            return -1

        a = a / np.linalg.norm(a)
        b = b / np.linalg.norm(b)

        return float(np.dot(a, b))

    def match(self, query_embedding, database_embeddings):

        if not database_embeddings:
            return {
                "matched": False,
                "employee_id": None,
                "name": None,
                "score": 0.0
            }

        best_score = -1.0
        best_match_name = None

        for employee in database_embeddings:

            name = employee["name"]
            db_embedding = employee["embedding"]

            score = self.cosine_similarity(query_embedding, db_embedding)

            if score > best_score:
                best_score = score
                best_match_name = name

        print(f"Best similarity score: {best_score:.4f}")

        if best_score >= self.threshold:
            return {
                "matched": True,
                "employee_id": None,
                "name": best_match_name,
                "score": best_score
            }
        else:
            return {
                "matched": False,
                "employee_id": None,
                "name": None,
                "score": best_score
            }