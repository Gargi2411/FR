import os
import cv2
import insightface
import numpy as np
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

    # Original
    augmented_images.append(img)

    # Brightness increase
    augmented_images.append(cv2.convertScaleAbs(img, alpha=1.2, beta=20))

    # Brightness decrease
    augmented_images.append(cv2.convertScaleAbs(img, alpha=0.9, beta=-10))

    # Slight blur
    augmented_images.append(cv2.GaussianBlur(img, (5, 5), 0))

    # Slight rotation +5
    h, w = img.shape[:2]
    M1 = cv2.getRotationMatrix2D((w//2, h//2), 5, 1)
    augmented_images.append(cv2.warpAffine(img, M1, (w, h)))

    # Slight rotation -5
    M2 = cv2.getRotationMatrix2D((w//2, h//2), -5, 1)
    augmented_images.append(cv2.warpAffine(img, M2, (w, h)))

    return augmented_images


def main():
    for person_name in os.listdir(IMAGE_FOLDER):
        person_folder = os.path.join(IMAGE_FOLDER, person_name)

        if not os.path.isdir(person_folder):
            continue

        print(f"Processing {person_name}...")

        all_embeddings = []

        for image_file in os.listdir(person_folder):
            image_path = os.path.join(person_folder, image_file)
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
                all_embeddings.append(embedding)

        if len(all_embeddings) == 0:
            print(f"No face found for {person_name}")
            continue

        # 🔥 Average embeddings
        mean_embedding = np.mean(all_embeddings, axis=0)

        # Normalize again after averaging
        mean_embedding = normalize_embedding(mean_embedding)

        # Store ONLY ONE strong embedding per person
        db.insert_employee(person_name, person_name, mean_embedding)

        print(f"{person_name} stored with {len(all_embeddings)} augmented embeddings.")

    print("Database build complete.")


if __name__ == "__main__":
    main()
