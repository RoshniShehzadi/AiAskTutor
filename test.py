# import torch
# x = torch.rand(5, 3)
# print(x)
# from sentence_transformers import SentenceTransformer

# try:
#     model = SentenceTransformer('all-MiniLM-L6-v2')
#     print("Model loaded successfully!")
# except Exception as e:
#     print(f"Error loading model: {e}")
import speech_recognition as sr

recognizer = sr.Recognizer()

# Specify the correct device (e.g., plughw:CARD=sofhdadsp,DEV=6)
mic = sr.Microphone(device_index=6)  # Use the correct device index

with mic as source:
    print("Say something...")
    audio = recognizer.listen(source)
    print("Recognizing...")
    try:
        text = recognizer.recognize_google(audio)
        print(f"Recognized text: {text}")
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand the audio")
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")



