import sys
import os
import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier

xgb_model = XGBClassifier()
xgb_model._estimator_type = "classifier" 
xgb_model.load_model('best_xgb_model.json')

le = joblib.load('label_encoder.pkl')

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

def preprocess_live_landmarks(landmarks_list):
    landmarks = np.array(landmarks_list).reshape(21, 3)
    wrist = landmarks[0]
    landmarks[:, 0] -= wrist[0]
    landmarks[:, 1] -= wrist[1]
    mid_finger_tip = landmarks[12]
    scale_factor = np.linalg.norm(mid_finger_tip[:2])
    if scale_factor > 0:
        landmarks[:, :2] /= scale_factor
    return landmarks.flatten()

cap = cv2.VideoCapture(0)
print("Starting Live Feed... Press 'q' to exit")

while cap.isOpened():
    success, image = cap.read()
    if not success: break

    image = cv2.flip(image, 1) 
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_image)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            raw_list = []
            for lm in hand_landmarks.landmark:
                raw_list.extend([lm.x * image.shape[1], lm.y * image.shape[0], lm.z])
            
            processed_landmarks = preprocess_live_landmarks(raw_list)
            df_live = pd.DataFrame([processed_landmarks])
            
            pred_numeric = xgb_model.predict(df_live)[0]
            prediction = le.inverse_transform([pred_numeric])[0]
            
            cv2.putText(image, f"Gesture: {prediction}", (10, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

    cv2.imshow('Hand Gesture Live Classification', image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()