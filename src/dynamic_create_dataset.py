import cv2
import mediapipe as mp
import numpy as np
import time
import os

def load_existing_actions(dataset_dir='dynamic_dataset'):
    """Load existing actions from actions.txt if it exists"""
    actions_file = os.path.join(dataset_dir, 'actions.txt')
    existing_actions = []
    if os.path.exists(actions_file):
        with open(actions_file, 'r') as f:
            existing_actions = f.read().strip().split('\n')
        print("\nExisting actions:", existing_actions)
    return existing_actions

def get_new_actions(existing_actions):
    while True:
        try:
            print("\nEnter new signs/words to add (separated by spaces):")
            new_actions = input().strip().split()
            if not new_actions:
                print("Error: No actions provided. Please try again.")
                continue
            
            # Check for duplicates
            duplicates = [action for action in new_actions if action in existing_actions]
            if duplicates:
                print(f"\nWarning: These actions already exist: {duplicates}")
                confirm = input("Do you want to record them again? (y/n): ").lower()
                if confirm != 'y':
                    new_actions = [action for action in new_actions if action not in duplicates]
                    if not new_actions:
                        print("No new actions to record. Please try again.")
                        continue
            
            print(f"\nWill record the following {len(new_actions)} actions:")
            for idx, action in enumerate(new_actions):
                print(f"{idx + 1}. {action}")
            
            confirm = input("\nConfirm these actions? (y/n): ").lower()
            if confirm == 'y':
                return new_actions
        except EOFError:
            print("\nError reading input. Please try again.")
        except KeyboardInterrupt:
            print("\nProgram interrupted by user.")
            exit()

def main():
    # Load existing actions
    existing_actions = load_existing_actions()
    
    # Get new actions to add
    new_actions = get_new_actions(existing_actions)
    
    # Calculate starting index for new actions
    start_idx = len(existing_actions)
    
    input("\nPress Enter to start recording...")

    seq_length = 30
    secs_for_action = 30

    # MediaPipe hands model
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=0)

    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return

    created_time = int(time.time())
    dataset_dir = 'dynamic_dataset'
    os.makedirs(dataset_dir, exist_ok=True)

    # Update actions.txt with new actions
    all_actions = existing_actions + new_actions
    with open(os.path.join(dataset_dir, 'actions.txt'), 'w') as f:
        f.write('\n'.join(all_actions))

    try:
        while cap.isOpened():
            for idx, action in enumerate(new_actions):
                actual_idx = start_idx + idx  # Use the correct index for the label
                data = []

                ret, img = cap.read()
                if not ret:
                    print("Error: Could not read frame")
                    continue

                img = cv2.flip(img, 1)

                # Create countdown
                for countdown in range(5, 0, -1):
                    display_img = img.copy()
                    cv2.putText(display_img, 
                              f'Preparing to record "{action.upper()}" in {countdown}...', 
                              org=(10, 30), 
                              fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                              fontScale=1, 
                              color=(255, 255, 255), 
                              thickness=2)
                    cv2.imshow('Recording', display_img)
                    cv2.waitKey(1000)

                start_time = time.time()
                frames_collected = 0

                while time.time() - start_time < secs_for_action:
                    ret, img = cap.read()
                    if not ret:
                        continue

                    img = cv2.flip(img, 1)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    result = hands.process(img)
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                    if result.multi_hand_landmarks is not None:
                        for res in result.multi_hand_landmarks:
                            joint = np.zeros((21, 4))
                            for j, lm in enumerate(res.landmark):
                                joint[j] = [lm.x, lm.y, lm.z, lm.visibility]

                            v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19], :3]
                            v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], :3]
                            v = v2 - v1
                            v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

                            angle = np.arccos(np.einsum('nt,nt->n',
                                v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:], 
                                v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) 

                            angle = np.degrees(angle)

                            d = np.concatenate([
                                joint.flatten(),
                                angle,
                                np.array([float(actual_idx)])  # Use the correct index
                            ])

                            data.append(d)
                            frames_collected += 1
                            mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

                    remaining_time = int(secs_for_action - (time.time() - start_time))
                    cv2.putText(img, 
                              f'Recording "{action.upper()}" - {remaining_time}s left ({frames_collected} frames)', 
                              org=(10, 30), 
                              fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                              fontScale=1, 
                              color=(255, 255, 255), 
                              thickness=2)
                    cv2.imshow('Recording', img)
                    
                    if cv2.waitKey(1) == ord('q'):
                        raise KeyboardInterrupt

                if data:
                    data = np.array(data)
                    print(f"\n{action}:")
                    print(f"Raw data shape: {data.shape}")
                    
                    raw_path = os.path.join(dataset_dir, f'raw_{action}_{created_time}')
                    np.save(raw_path, data)
                    
                    full_seq_data = []
                    for seq in range(len(data) - seq_length):
                        full_seq_data.append(data[seq:seq + seq_length])

                    full_seq_data = np.array(full_seq_data)
                    print(f"Sequence data shape: {full_seq_data.shape}")
                    
                    unique_labels = np.unique(full_seq_data[:, 0, -1])
                    print(f"Labels in data: {unique_labels}")
                    
                    seq_path = os.path.join(dataset_dir, f'seq_{action}_{created_time}')
                    np.save(seq_path, full_seq_data)
                else:
                    print(f"\nWarning: No data collected for {action}")
                
                display_img = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(display_img, 
                          f'Completed recording "{action}"', 
                          org=(10, 30), 
                          fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                          fontScale=1, 
                          color=(255, 255, 255), 
                          thickness=2)
                
                if idx < len(new_actions) - 1:
                    cv2.putText(display_img, 
                              'Press any key to continue to next action...', 
                              org=(10, 70), 
                              fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                              fontScale=1, 
                              color=(255, 255, 255), 
                              thickness=2)
                    cv2.imshow('Recording', display_img)
                    cv2.waitKey(0)
                else:
                    cv2.putText(display_img, 
                              'Recording completed! Press any key to exit...', 
                              org=(10, 70), 
                              fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                              fontScale=1, 
                              color=(255, 255, 255), 
                              thickness=2)
                    cv2.imshow('Recording', display_img)
                    cv2.waitKey(0)

            break

    except KeyboardInterrupt:
        print("\nRecording interrupted by user")
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()