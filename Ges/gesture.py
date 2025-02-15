import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
import google.generativeai as palm

# Configure Palm API
palm.configure(api_key="AIzaSyBOkwlZhUwyVdoKivghihp-l31rjSAaIJs")

# Initialize MediaPipe Hand Detection
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Gesture Dictionary
GESTURES = {
    "thumbs_up": "Thumbs Up 👍",
    "thumbs_down": "Thumbs Down 👎",
    "hi": "Hi (Waving) 👋",
    "good": "Good (Thumbs Up with Bent Wrist)",
    "bad": "Bad (Thumbs Down with Bent Wrist)",
    "click": "Click (Index & Thumb Touching) 🖱️",
    "select": "Select (Fist with Index Finger Extended) ✋",
    "open_hand": "Open Hand (Scroll) 🖐️",
    "pointing": "Pointing (Highlight) ☝️",
    "fist": "Fist (Select)",
    "peace_sign": "Peace Sign",
    "okay_sign": "Okay Sign",
    "palm_down": "Palm Down",
    "rock_on": "Rock On",

    "unknown": "Unknown Gesture ❓"
}

# Function to classify hand gestures
def classify_gesture(landmarks):
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    middle_tip = landmarks[12]
    ring_tip = landmarks[16]
    pinky_tip = landmarks[20]

    # Thumb Up
    if thumb_tip.y < index_tip.y and thumb_tip.y < middle_tip.y:
        return "thumbs_up"
    
    # Thumb Down
    elif thumb_tip.y > index_tip.y and thumb_tip.y > middle_tip.y:
        return "thumbs_down"

    # Waving Hand (Hi)
    elif (
        index_tip.y < middle_tip.y < ring_tip.y < pinky_tip.y and
        abs(index_tip.x - pinky_tip.x) > 0.1  # Hand spread wide
    ):
        return "hi"

    # Good - Thumbs Up but wrist bent
    elif thumb_tip.y < index_tip.y and abs(thumb_tip.x - index_tip.x) > 0.05:
        return "good"

    # Bad - Thumbs Down but wrist bent
    elif thumb_tip.y > index_tip.y and abs(thumb_tip.x - index_tip.x) > 0.05:
        return "bad"

    # Click - Index and Thumb Touching
    elif abs(index_tip.x - thumb_tip.x) < 0.02 and abs(index_tip.y - thumb_tip.y) < 0.02:
        return "click"

    # Select - Fist but index finger extended
    elif (
        index_tip.y < middle_tip.y and
        middle_tip.y > ring_tip.y and ring_tip.y > pinky_tip.y
    ):
        return "select"

    # Open Hand
    elif (
        index_tip.y < middle_tip.y < ring_tip.y < pinky_tip.y and
        abs(index_tip.x - pinky_tip.x) > 0.05
    ):
        return "open_hand"

    # Pointing - Index Finger Extended
    elif index_tip.x < thumb_tip.x and middle_tip.y > ring_tip.y:
        return "pointing"

    # Fist - All fingers curled
    elif (
        index_tip.y > landmarks[5].y and
        middle_tip.y > landmarks[9].y and
        ring_tip.y > landmarks[13].y and
        pinky_tip.y > landmarks[17].y
    ):
        return "fist"
     # Peace Sign (index and middle fingers up)
    elif (
        index_tip.y < landmarks[6].y
        and middle_tip.y < landmarks[10].y
        and thumb_tip.y > landmarks[2].y
    ):
        return "peace_sign"
    
    # Okay Sign (thumb and index form a circle)
    elif (
        abs(thumb_tip.x - index_tip.x) < 0.05
        and abs(thumb_tip.y - index_tip.y) < 0.05
    ):
        return "okay_sign"
    
    # Palm Down
    elif thumb_tip.y < index_tip.y and middle_tip.y < index_tip.y and ring_tip.y < index_tip.y:
        return "palm_down"
    
    # Rock On (index and pinky up, other fingers curled)
    elif (
        index_tip.y < landmarks[5].y
        and pinky_tip.y < landmarks[17].y
        and middle_tip.y > landmarks[9].y
        and ring_tip.y > landmarks[13].y
    ):
        return "rock_on"


    return "unknown"

# Function to get AI description from Palm
def get_ai_description(gesture_name):
    model = palm.GenerativeModel("gemini-pro")  # Use Gemini models
    response = model.generate_content(f"Describe the meaning and usage of the gesture: {gesture_name}")
    
    return response.text if response and response.text else "No description available."

# Streamlit Interface
st.title("Gesture-Based Human-Computer Interaction System")
st.write("Control the interface using hand gestures in real-time.")

# Video Capture
cap = cv2.VideoCapture(0)
frame_placeholder = st.empty()
gesture_text = st.empty()
ai_description = st.empty()

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    gesture_name = "unknown"

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            landmarks = hand_landmarks.landmark
            gesture_name = classify_gesture(landmarks)

    # Display Video Feed
    frame_placeholder.image(frame, channels="BGR", use_container_width=True)
    gesture_text.write(f"*Recognized Gesture:* {GESTURES.get(gesture_name, 'Unknown')}")

    # Get AI-based description
    if gesture_name != "unknown":
        description = get_ai_description(GESTURES[gesture_name])
        ai_description.write(f"*AI Description:* {description}")

cap.release()
cv2.destroyAllWindows()
