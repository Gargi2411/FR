import os
import cv2
import insightface
from src.database.db_manager import DatabaseManager
from src.matching.matcher import Matcher

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "data/embeddings.db")

app = insightface.app.FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=0)

db = DatabaseManager(DB_PATH)
matcher = Matcher(threshold=0.6)

def main():
    img_path = os.path.join(BASE_DIR, "data/raw_images/test3.jpg")
    img = cv2.imread(img_path)

    if img is None:
        print("Image not found")
        return

    faces = app.get(img)
    database_embeddings = db.get_all_embeddings()

    for face in faces:
        embedding = face.embedding
        result = matcher.match(embedding, database_embeddings)

        x1, y1, x2, y2 = map(int, face.bbox)

        if result["matched"]:
            label = result["name"]
            color = (0, 255, 0)  # Green
        else:
            label = "Unknown"
            color = (0, 0, 255)  # Red

        # Draw box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # Draw name
        cv2.putText(
            img,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2
        )

    cv2.imshow("Face Recognition", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()