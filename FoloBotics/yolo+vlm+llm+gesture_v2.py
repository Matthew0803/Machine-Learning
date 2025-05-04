#pip install opencv-python mediapipe transformers torch torchvision playsound openai
import cv2
import time
from PIL import Image
from transformers import pipeline, AutoImageProcessor
from ultralytics import YOLO
import mediapipe as mp
from openai import OpenAI  # Added OpenAI import

TOKEN = "YOUR_HUGGINGFACE_TOKEN"  # Replace with your Hugging Face token
OPENAI_API_KEY = "YOUR_OPENAI_API_KEY"  # Add your OpenAI key

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

# Initialize OpenAI client
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# Initialize models
person_detector = YOLO('yolov10n.pt')
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)

# Vision-Language pipeline
image_processor = AutoImageProcessor.from_pretrained("Salesforce/blip-image-captioning-base", use_fast=True)
ImageToText = pipeline(
    "image-to-text",
    model="Salesforce/blip-image-captioning-base",
    image_processor=image_processor,
    device="cuda",
    framework="pt"
)

# Removed Hugging Face text-generation pipeline

# State machine
class ConversationState:
    def __init__(self):
        self.states = {
            "WAITING": 0,
            "IN_CONVERSATION": 1,
            "AWAITING_GESTURE": 2
        }
        self.current_state = self.states["WAITING"]
        self.last_gesture = None
        self.gesture_timeout = 5  # seconds
        self.last_activity = time.time()

# Gesture recognition helpers (unchanged)
def is_thumbs_up(landmarks):
    thumb_tip = landmarks.landmark[4]
    thumb_ip = landmarks.landmark[3]
    return thumb_tip.y < thumb_ip.y  # Thumb tip above joint

def is_thumbs_down(landmarks):
    thumb_tip = landmarks.landmark[4]
    thumb_ip = landmarks.landmark[3]
    return thumb_tip.y > thumb_ip.y  # Thumb tip below joint

# New OpenAI response generator
def generate_openai_response(messages, max_tokens=60):
    try:
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"OpenAI Error: {str(e)}")
        return "I'm having trouble responding right now."

# Main loop (modified response generation)
cam = cv2.VideoCapture(0)
state = ConversationState()
last_response = "Wave to start conversation!"
conversation_history = []

while True:
    ret, frame = cam.read()
    if not ret:
        break

    # State transitions
    current_time = time.time()
    if state.current_state == state.states["WAITING"]:
        # Person detection (unchanged)
        detections = person_detector(frame, verbose=False)[0]
        if any(box.cls == 0 for box in detections.boxes):
            state.current_state = state.states["IN_CONVERSATION"]
            state.last_activity = current_time

    elif state.current_state == state.states["IN_CONVERSATION"]:
        # Generate initial response
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_frame = Image.fromarray(rgb_frame)
        
        try:
            vlm_response = ImageToText(pil_frame)
            description = vlm_response[0]["generated_text"] if vlm_response else "a person"
            
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"say hi to: {description}"}
            ]
            
            # Replaced with OpenAI call
            last_response = generate_openai_response(messages)
            print(last_response)
            conversation_history.append(last_response)
            
            state.current_state = state.states["AWAITING_GESTURE"]
            state.last_activity = current_time

        except Exception as e:
            last_response = f"Error: {str(e)}"

    elif state.current_state == state.states["AWAITING_GESTURE"]:
        # Gesture detection
        results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        if results.multi_hand_landmarks:
            landmarks = results.multi_hand_landmarks[0]
            
            if is_thumbs_up(landmarks):
                # Continue conversation
                messages = [
                    {"role": "system", "content": "Continue conversation positively."},
                    #{"role": "assistant", "content": last_response},
                    {"role": "user", "content": "What else can you tell me?"}
                ]
                
                last_response = generate_openai_response(messages, max_tokens=40)
                print(last_response)
                state.current_state = state.states["IN_CONVERSATION"]
                
            elif is_thumbs_down(landmarks):
                # Comfort response
                messages = [
                    {"role": "system", "content": "Comfort the user gently."},
                    #{"role": "assistant", "content": last_response},
                    {"role": "user", "content": "I'm not happy today."}
                ]
                
                last_response = generate_openai_response(messages, max_tokens=40)
                print(last_response)
                state.current_state = state.states["WAITING"]

        # Timeout handling (unchanged)
        if current_time - state.last_activity > state.gesture_timeout:
            state.current_state = state.states["WAITING"]
            last_response = "Conversation timed out"

    # Display overlay (unchanged)
    display_text = {
        state.states["WAITING"]: "Wave to start conversation!",
        state.states["IN_CONVERSATION"]: "Processing...",
        state.states["AWAITING_GESTURE"]: "Show thumbs up/down"
    }[state.current_state]
    
    cv2.putText(frame, display_text, (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, last_response, (10, 70), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    cv2.imshow('Gesture-Controlled Conversation', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()