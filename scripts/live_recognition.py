import cv2
import insightface
from src.database.db_manager import DatabaseManager
from src.matching.matcher import Matcher
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "data/embeddings.db")

app = insightface.app.FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=0)

db = DatabaseManager(DB_PATH)
matcher = Matcher(threshold=0.6)

cap = cv2.VideoCapture(0)  # Laptop camera

print("Press Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    faces = app.get(frame)
    database_embeddings = db.get_all_embeddings()

    for face in faces:
        embedding = face.embedding
        result = matcher.match(embedding, database_embeddings)

        x1, y1, x2, y2 = map(int, face.bbox)

        if result["matched"]:
            label = result["name"]
            color = (0, 255, 0)
        else:
            label = "Unknown"
            color = (0, 0, 255)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("Live Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()