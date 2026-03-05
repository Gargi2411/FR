import cv2
import insightface

app = insightface.app.FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=0)

img = cv2.imread("data/raw_images/test.jpg")

faces = app.get(img)

if len(faces) == 0:
    print("No face detected")
else:
    face = faces[0]
    embedding = face.embedding
    print("Embedding shape:", embedding.shape)