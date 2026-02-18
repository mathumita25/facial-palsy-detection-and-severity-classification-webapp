import cv2
import numpy as np
import dlib
import pickle

# -------------------- Load models once (IMPORTANT) --------------------

face_net = cv2.dnn.readNetFromCaffe(
    "deploy.prototxt",
    "res10_300x300_ssd_iter_140000.caffemodel"
)

landmark_predictor = dlib.shape_predictor(
    "shape_predictor_68_face_landmarks.dat"
)

with open("facial_palsy_rf_model_equal_2000perclass.pkl", "rb") as f:
    model = pickle.load(f)

with open("model_info_equal_2000perclass.pkl", "rb") as f:
    model_info = pickle.load(f)

CLASS_NAMES = model_info["class_names"]

# -------------------- Helper functions --------------------

def detect_face_dnn(image, confidence_threshold=0.5):
    h, w = image.shape[:2]
    blob = cv2.dnn.blobFromImage(
        cv2.resize(image, (300, 300)),
        1.0,
        (300, 300),
        (104.0, 177.0, 123.0)
    )
    face_net.setInput(blob)
    detections = face_net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > confidence_threshold:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)
            return dlib.rectangle(x1, y1, x2, y2)

    raise ValueError("No face detected")

def get_landmarks(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face = detect_face_dnn(image)
    shape = landmark_predictor(gray, face)
    return np.array([[p.x, p.y] for p in shape.parts()])

def find_midline(landmarks):
    return (landmarks[30][0] + landmarks[8][0]) / 2

def create_clusters(landmarks, midline_x):
    clusters = {
        "left_eyebrow": [], "right_eyebrow": [],
        "left_eye": [], "right_eye": [],
        "left_mouth": [], "right_mouth": []
    }

    for i, (x, y) in enumerate(landmarks):
        if 17 <= i <= 26:
            (clusters["left_eyebrow"] if x < midline_x else clusters["right_eyebrow"]).append((x, y))
        elif 36 <= i <= 47:
            (clusters["left_eye"] if x < midline_x else clusters["right_eye"]).append((x, y))
        elif 48 <= i <= 67:
            (clusters["left_mouth"] if x < midline_x else clusters["right_mouth"]).append((x, y))

    return {k: np.array(v) for k, v in clusters.items()}

def angle(p1, p2):
    return np.degrees(np.arctan2(p2[1] - p1[1], p2[0] - p1[0]))

def extract_features(landmarks):
    mid = find_midline(landmarks)
    clusters = create_clusters(landmarks, mid)

    eyebrow_angle = angle(
        clusters["left_eyebrow"][0],
        clusters["right_eyebrow"][-1]
    )

    eye_angle = angle(
        clusters["left_eye"][0],
        clusters["right_eye"][3]
    )

    mouth_angle = angle(landmarks[48], landmarks[54])

    return {
        "eyebrow_angle": eyebrow_angle,
        "eye_angle": eye_angle,
        "mouth_angle": mouth_angle
    }

# -------------------- Public inference API --------------------

def predict_image(image: np.ndarray):
    landmarks = get_landmarks(image)
    features = extract_features(landmarks)

    X = np.array([[
        features["eyebrow_angle"],
        features["eye_angle"],
        features["mouth_angle"]
    ]])

    pred = int(model.predict(X)[0])
    probs = model.predict_proba(X)[0]

    return {
        "prediction": CLASS_NAMES[pred],
        "confidence": float(probs[pred]),
        "features": features
    }
