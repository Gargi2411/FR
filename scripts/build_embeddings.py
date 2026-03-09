import os
import cv2
import insightface
import numpy as np
from collections import defaultdict
from src.database.db_manager import DatabaseManager

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGE_FOLDER = os.path.join(BASE_DIR, "data/raw_images")
DB_PATH = os.path.join(BASE_DIR, "data/embeddings.db")

app = insightface.app.FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=0, det_size=(640, 640))

db = DatabaseManager(DB_PATH)


def normalize_embedding(embedding):
    return embedding / np.linalg.norm(embedding)


def augment_image(img):
    augmented_images = []

    augmented_images.append(img)

    augmented_images.append(cv2.convertScaleAbs(img, alpha=1.2, beta=20))
    augmented_images.append(cv2.convertScaleAbs(img, alpha=0.9, beta=-10))

    augmented_images.append(cv2.GaussianBlur(img, (5, 5), 0))

    h, w = img.shape[:2]

    M1 = cv2.getRotationMatrix2D((w//2, h//2), 5, 1)
    augmented_images.append(cv2.warpAffine(img, M1, (w, h)))

    M2 = cv2.getRotationMatrix2D((w//2, h//2), -5, 1)
    augmented_images.append(cv2.warpAffine(img, M2, (w, h)))

    return augmented_images


def extract_name_from_filename(filename):
    name = os.path.splitext(filename)[0]
    name = ''.join([c for c in name if not c.isdigit()])
    name = name.replace("_", " ").strip()
    return name


def main():

    # 🔥 Get employees already stored in DB
    existing_employees = set()

    all_db_embeddings = db.get_all_embeddings()

    for emp in all_db_embeddings:
        existing_employees.add(emp["name"])

    print("Employees already in DB:", existing_employees)

    employee_embeddings = defaultdict(list)

    for image_file in os.listdir(IMAGE_FOLDER):

        if not image_file.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        person_name = extract_name_from_filename(image_file)

        # 🔥 Skip if already exists
        if person_name in existing_employees:
            print(f"Skipping {person_name} (already in DB)")
            continue

        image_path = os.path.join(IMAGE_FOLDER, image_file)

        print(f"Processing {image_file} → {person_name}")

        img = cv2.imread(image_path)

        if img is None:
            continue

        augmented_images = augment_image(img)

        for aug_img in augmented_images:

            faces = app.get(aug_img)

            if len(faces) == 0:
                continue

            embedding = faces[0].embedding
            embedding = normalize_embedding(embedding)

            employee_embeddings[person_name].append(embedding)

    for person_name, embeddings in employee_embeddings.items():

        if len(embeddings) == 0:
            print(f"No face found for {person_name}")
            continue

        mean_embedding = np.mean(embeddings, axis=0)
        mean_embedding = normalize_embedding(mean_embedding)

        db.insert_employee(person_name, mean_embedding)

        print(f"{person_name} added to DB with {len(embeddings)} embeddings.")

    print("Database update complete.")


if __name__ == "__main__":
    main()