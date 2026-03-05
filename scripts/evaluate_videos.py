import cv2
import insightface
import os
import numpy as np
from src.database.db_manager import DatabaseManager
from src.matching.matcher import Matcher

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_FOLDER = os.path.join(BASE_DIR, "data/test_videos")
DB_PATH = os.path.join(BASE_DIR, "data/embeddings.db")

# Load model
app = insightface.app.FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=0, det_size=(640, 640))

db = DatabaseManager(DB_PATH)
matcher = Matcher(threshold=0.45)

database_embeddings = db.get_all_embeddings()


def draw_prediction(frame, result, face):
    x1, y1, x2, y2 = map(int, face.bbox)

    if result["matched"]:
        label = f"{result['name']} ({result['score']:.2f})"
        color = (0, 255, 0)
    else:
        label = "Unknown"
        color = (0, 0, 255)

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def main():

    person_folders = [
        f for f in os.listdir(TEST_FOLDER)
        if os.path.isdir(os.path.join(TEST_FOLDER, f))
    ]

    if not person_folders:
        print("No person folders found.")
        return

    person_name = person_folders[0]
    person_folder = os.path.join(TEST_FOLDER, person_name)

    video_paths = [
        os.path.join(person_folder, v)
        for v in os.listdir(person_folder)
        if v.endswith(".mp4")
    ]

    caps = [cv2.VideoCapture(v) for v in video_paths]

    print("Press Q to quit")

    while True:

        frames = []

        for cap in caps:
            ret, frame = cap.read()

            if not ret:
                blank = np.zeros((360, 480, 3), dtype=np.uint8)
                frames.append(blank)
                continue

            faces = app.get(frame)

            for face in faces:
                if face.det_score < 0.6:
                    continue

                embedding = face.embedding
                result = matcher.match(embedding, database_embeddings)

                draw_prediction(frame, result, face)

            frame = cv2.resize(frame, (480, 360))
            frames.append(frame)

        if len(frames) == 0:
            break

        # ---- AUTO GRID FIX ----
        resized_frames = frames

        # If odd number of videos, add black frame
        if len(resized_frames) % 2 != 0:
            blank = np.zeros((360, 480, 3), dtype=np.uint8)
            resized_frames.append(blank)

        rows = []
        for i in range(0, len(resized_frames), 2):
            row = np.hstack(resized_frames[i:i+2])
            rows.append(row)

        combined = np.vstack(rows)
        # ------------------------

        cv2.imshow("Live Multi-Video Face Recognition", combined)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    for cap in caps:
        cap.release()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()