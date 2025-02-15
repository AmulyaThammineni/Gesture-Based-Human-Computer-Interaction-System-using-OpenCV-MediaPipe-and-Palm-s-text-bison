
# Gesture-Based Human-Computer Interaction System

This project is a real-time hand gesture recognition system that allows users to control an interface using hand gestures. It uses OpenCV, MediaPipe, and Google AI Studio's Gemini models to recognize and describe hand gestures.

## Features
- Real-time hand gesture detection using a webcam.
- Classification of common hand gestures.
- AI-generated descriptions of gestures using Google AI Studio.

## Installation

### 1. Clone the repository

git clone https://github.com/AmulyaThammineni/Gesture-Based-Human-Computer-Interaction-System-using-OpenCV-MediaPipe-and-Palm-s-text-bison.git

### 2. Install dependencies

pip install -r requirements.txt


### 3. Run the application

streamlit run gesture.py



## Setting Up Google AI Studio API Key

To use the Gemini AI model for generating gesture descriptions, you need to obtain an API key from Google AI Studio.

### Steps to Generate API Key:

1. Go to [Google AI Studio](https://aistudio.google.com/)
2. Sign in with your Google account.
3. Navigate to the API Keys section.
4. Click Generate API Key.
5. Copy the generated API key.
6. Replace the placeholder in the code:
   
   palm.configure(api_key="YOUR_API_KEY_HERE")

7. Save the file and run the application.


