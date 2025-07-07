import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
import os

# MediaPipe hands model - updated initialization
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# Load your trained model
model = load_model('models/best_model.keras')

# Get the number of actions from the model's output shape
num_classes = model.output_shape[-1]

# Get list of actions from sequence files in dataset directory
sequence_files = [f for f in os.listdir('dynamic_dataset') if f.startswith('seq_')]

actions_file = os.path.join('dynamic_dataset', 'actions.txt')
if os.path.exists(actions_file):
    with open(actions_file, 'r') as f:
        actions = f.read().strip().split('\n')

print(f"Detected actions: {actions}")
print(f"Number of classes: {num_classes}")

if len(actions) != num_classes:
    raise ValueError(f"Number of detected actions ({len(actions)}) doesn't match model output ({num_classes})")

seq_length = 15
# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera")
    exit()

# Initialize sequence storage
seq = []
action_seq = []

def draw_text_with_background(img, text, position, scale=1, thickness=2):
    """Helper function to draw text with background"""
    font = cv2.FONT_HERSHEY_SIMPLEX
    # Get text size
    size = cv2.getTextSize(text, font, scale, thickness)[0]
    
    # Draw background rectangle
    x, y = position
    cv2.rectangle(img, (x-10, y-size[1]-10), (x+size[0]+10, y+10), (0,0,0), -1)
    
    # Draw text
    cv2.putText(img, text, (x, y), font, scale, (255,255,255), thickness)

while True:
    # Read frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame")
        break

    # Flip and process frame
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process hands
    results = hands.process(frame_rgb)
    
    # Draw hand landmarks
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Extract joint coordinates
            joint = np.zeros((21, 4))
            for j, lm in enumerate(hand_landmarks.landmark):
                joint[j] = [lm.x, lm.y, lm.z, lm.visibility]

            # Compute angles between joints
            v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19], :3]
            v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], :3]
            v = v2 - v1
            
            # Normalize vectors
            v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

            # Calculate angles
            angle = np.arccos(np.einsum('nt,nt->n',
                v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:], 
                v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:]))
            angle = np.degrees(angle)

            # Create feature array
            d = np.concatenate([joint.flatten(), angle])
            seq.append(d)

            # Draw landmarks
            mp_drawing.draw_landmarks(
                frame, 
                hand_landmarks, 
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
            )

            # Make prediction when we have enough frames
            if len(seq) < seq_length:
                continue

            # Prepare input data
            input_data = np.expand_dims(np.array(seq[-seq_length:], dtype=np.float32), axis=0)
            
            # Get prediction
            y_pred = model.predict(input_data, verbose=0)[0]
            predicted_idx = int(np.argmax(y_pred))
            confidence = y_pred[predicted_idx]

            # Only proceed if confidence is high enough
            if confidence < 0.95:
                continue

            # Show only the top prediction with highest confidence
            action = actions[predicted_idx]
           
            # Display the top prediction
            draw_text_with_background(
                frame,
                f'{action}: {confidence:.2f}',
                (10, 30)
            )
            # Optional: Print top prediction to console for debugging
            print(f"Top prediction: {action}, confidence: {confidence:.2f}")

            action_seq.append(action)

            # Only show prediction after 2 consistent predictions
            if len(action_seq) < 2:
                continue


            # Keep sequences from getting too long
            if len(seq) > seq_length:
                seq = seq[-seq_length:]
            if len(action_seq) > 5:
                action_seq = action_seq[-5:]

    # Display help text
    draw_text_with_background(frame, "Press 'q' to quit", (10, frame.shape[0] - 20))
    
    # Show frame
    cv2.imshow('Hand Gesture Recognition', frame)
    
    # Check for quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()