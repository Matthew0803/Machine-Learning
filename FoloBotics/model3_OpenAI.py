#pip install openai transformers torch torchvision playsound scipy opencv-python Pillow requests
import cv2
import os
import torch
import numpy as np
import scipy.io.wavfile
import threading
from queue import Queue
from transformers import pipeline
from openai import OpenAI
from playsound import playsound
import tempfile
from PIL import Image
import signal

running = True
processing_lock = threading.Lock()
audio_ready = threading.Event()
audio_ready.set()

def signal_handler(sig, frame):
    global running
    print("\nShutting down gracefully...")
    running = False

signal.signal(signal.SIGINT, signal_handler)

# Configuration
TOKEN = "YOUR_HUGGINGFACE_API_KEY"  # Replace with your Hugging Face API key
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --------------------------
# Improved Prompt Engineering
# --------------------------
SYSTEM_PROMPT = """You are a friendly and observant conversational assistant. 
When greeting users, follow this structure:
1. Warm greeting with appropriate title (sir/madam/friend)
2. Comment on visible elements from the image
3. Add a cheerful observation about the environment
4. End with an open-ended question
"""

# --------------------------
# Audio System Initialization (Add at the top)
# --------------------------
audio_queue = Queue()
audio_lock = threading.Lock()

# --------------------------
# Audio Queue System
# --------------------------
def audio_worker():
    while running:
        try:
            file_path = audio_queue.get(timeout=1)
            with audio_lock:
                if running:
                    playsound(file_path)
            os.remove(file_path)
            audio_queue.task_done()
            audio_ready.set()  # Signal that audio is done
        except Exception as e:
            if running:
                continue
            else:
                break

# Start audio thread
threading.Thread(target=audio_worker, daemon=True).start()

# --------------------------
# Model Loading (Outside Loop)
# --------------------------
def load_models():
    """Load all models once at startup"""
    models = {}
    
    # Image captioning
    models['image_pipe'] = pipeline(
        "image-to-text",
        model="Salesforce/blip-image-captioning-base",
        device=DEVICE
    )
    
    # Text-to-speech (MMS-TTS doesn't need forward_params)
    models['tts_pipe'] = pipeline(
        "text-to-speech",
        model="facebook/mms-tts-eng",
        device=DEVICE
    )

    # Warmup models with proper formats
    dummy_image = Image.new('RGB', (224, 224), color='red')
    _ = models['image_pipe'](dummy_image)
    
    # Warmup TTS without do_sample
    _ = models['tts_pipe']("Warmup")  # Remove forward_params
    
    return models

# Initialize models once
try:
    models = load_models()
except Exception as e:
    print(f"Initialization error: {str(e)}")
    exit()
# --------------------------
# OpenAI API Setup
# --------------------------
OPENAI_API_KEY = "YOUR_OPENAI_API_KEY"
client = OpenAI(
    api_key=OPENAI_API_KEY
)  
def generate(messages):
    try:
        response = client.responses.create(
            model="gpt-4", # or "gpt-4"
            instructions = messages[0],
            input = messages[1],
            max_output_tokens=128,
            temperature=0.7,
            top_p=0.9,
        )
        return response.output_text
    
    except Exception as e:
        print(f"OpenAI API Error: {str(e)}")
        return "I'm having trouble responding right now."

# --------------------------
# Processing Function
# --------------------------
def process_frame(frame, count, models):
    global processing_lock
    if not running: 
        return  # Check global flag
    with processing_lock:
        try:
            # Image captioning
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_frame = Image.fromarray(rgb_frame)
            caption_result = models['image_pipe'](pil_frame)
            image_description = caption_result[0]["generated_text"]

            # LLM prompt with structured format
            messages = [
                SYSTEM_PROMPT,
                "Create a friendly greeting based on: " + image_description
            ]
            
            # LLM response
            text_response = generate(messages)

            if not text_response.strip():
                raise ValueError("Empty response from API")
            
            if '"' in text_response:
                start = text_response.find('"') + 1
                end = text_response.find('"', start)
                clean_response = text_response[start:end]
            else:
                clean_response = text_response.split("\n")[-1].strip()
            
            # Clean response
            print(clean_response)

            # Text to speech with temp file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
                speech = models['tts_pipe'](clean_response)
                audio_array = np.array(speech["audio"]).squeeze()
                scipy.io.wavfile.write(tf.name, speech["sampling_rate"], audio_array)
                audio_queue.put(tf.name)  # Add to playback queue

        except Exception as e:
            print(f"Processing error: {str(e)}")

# --------------------------
# Main Loop
# --------------------------
cam = cv2.VideoCapture(0)
#clear_frames()  # Assume this is defined as before
count = 0
frame_interval = 500

try:
    while running:
        ret, frame = cam.read()
        if not ret: break

        if count % frame_interval == 0 and audio_ready.is_set() and audio_queue.empty():
            frame_thread = threading.Thread(
                target=process_frame, 
                args=(frame.copy(),count, models)
                )
            frame_thread.start()
        
        cv2.imshow('Camera', frame)
        if cv2.waitKey(1) == ord('q'): break
        
        count += 1
finally:
    # Cleanup sequence
    print("Cleaning up resources...")
    running = False
    
    # Release camera
    cam.release()
    cv2.destroyAllWindows()
    
    # Clear audio queue
    while not audio_queue.empty():
        try:
            file_path = audio_queue.get_nowait()
            os.remove(file_path)
        except:
            pass
audio_queue.join()
print("Shutdown complete")


cam.release()
cv2.destroyAllWindows()
