import cv2
import insightface
import os
from collections import Counter

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


def evaluate_video(video_path, true_person):

    cap = cv2.VideoCapture(video_path)

    predictions = []

    while True:

        ret, frame = cap.read()
        if not ret:
            break

        faces = app.get(frame)

        for face in faces:

            if face.det_score < 0.6:
                continue

            embedding = face.embedding
            result = matcher.match(embedding, database_embeddings)

            if result["matched"]:
                predictions.append(result["name"])
            else:
                predictions.append("Unknown")

    cap.release()

    if len(predictions) == 0:
        return False

    majority_prediction = Counter(predictions).most_common(1)[0][0]

    print(f"Video: {os.path.basename(video_path)}")
    print("Predicted:", majority_prediction)
    print("Actual:", true_person)
    print()

    return majority_prediction == true_person


def main():

    total_videos = 0
    correct_videos = 0

    person_folders = [
        f for f in os.listdir(TEST_FOLDER)
        if os.path.isdir(os.path.join(TEST_FOLDER, f))
    ]

    for person in person_folders:

        person_path = os.path.join(TEST_FOLDER, person)

        videos = [
            v for v in os.listdir(person_path)
            if v.endswith(".mp4")
        ]

        for video in videos:

            video_path = os.path.join(person_path, video)

            total_videos += 1

            correct = evaluate_video(video_path, person)

            if correct:
                correct_videos += 1

    print("====== FINAL RESULTS ======")
    print("Total Videos:", total_videos)
    print("Correct Predictions:", correct_videos)

    accuracy = (correct_videos / total_videos) * 100
    print("Accuracy:", round(accuracy, 2), "%")


if __name__ == "__main__":
    main()