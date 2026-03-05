import cv2
import insightface
import os
import time
import numpy as np
from src.database.db_manager import DatabaseManager
from src.matching.matcher import Matcher

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIDEO_PATH = os.path.join(BASE_DIR, "data/test_videos/test.mp4")
DB_PATH = os.path.join(BASE_DIR, "data/embeddings.db")

# ✅ Use stable InsightFace model
app = insightface.app.FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=0, det_size=(640, 640))  # 1024 not needed unless faces very small

db = DatabaseManager(DB_PATH)
matcher = Matcher(threshold=0.45)


def enhance_frame(frame):
    """
    Mild brightness correction without destroying natural features.
    """
    return cv2.convertScaleAbs(frame, alpha=1.1, beta=10)


def main():
    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print("Error opening video file")
        return

    print("Processing video... Press Q to quit")

    database_embeddings = db.get_all_embeddings()

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # ✅ Process every 2nd frame (improves stability + speed)
        if frame_count % 2 != 0:
            continue

        frame = enhance_frame(frame)

        faces = app.get(frame)

        for face in faces:

            # ✅ Skip weak detections
            if face.det_score < 0.6:
                continue

            embedding = face.embedding

            # ✅ Measure matching time
            start = time.time()
            result = matcher.match(embedding, database_embeddings)
            match_time = time.time() - start
            print("Match time:", round(match_time, 4), "sec")

            x1, y1, x2, y2 = map(int, face.bbox)

            if result["matched"]:
                label = f"{result['name']} ({result['score']:.2f})"
                color = (0, 255, 0)
            else:
                label = "Unknown"
                color = (0, 0, 255)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.imshow("Video Face Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()