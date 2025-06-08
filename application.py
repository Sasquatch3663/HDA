import cv2
import mediapipe as mp

# Initialize mediapipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Create Hands object
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,    min_tracking_confidence=0.7
)

# Open webcamcap = cv2.VideoCapture(0)

# Determine finger states
def get_finger_states(hand_landmarks, handedness_label):
    finger_states = []

    # Thumb logic adjusted for left vs right
    if handedness_label == 'Right':
        thumb_open = hand_landmarks.landmark[4].x > hand_landmarks.landmark[3].x
    else:
        thumb_open = hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x
    finger_states.append(thumb_open)

    # Other fingers
    tips = [8, 12, 16, 20]
    pips = [6, 10, 14, 18]
    for tip, pip in zip(tips, pips):
        finger_states.append(hand_landmarks.landmark[tip].y < hand_landmarks.landmark[pip].y)

    return finger_states

# Gesture classification
def classify_gesture(states):
    gestures = {
        (False, False, False, False, False): "Fist",
        (True, True, True, True, True): "Open Hand",
        (False, True, False, False, False): "One",
        (False, True, True, False, False): "Two",
        (False, True, True, True, False): "Three",
        (False, True, True, True, True): "Four",
        (True, True, True, True, True): "Five",
        (True, False, False, False, False): "Thumbs Up",
        (False, False, False, False, True): "Pinky Only",
        (True, False, False, False, True): "Call Me 🤙",
        (True, True, False, False, True): "Rock 🤘",
        (False, False, True, True, False): "Middle & Ring",
        (False, True, False, False, True): "ILY ❤️",
        (False, True, False, False, False): "ASL: D",
        (True, False, False, True, True): "ASL: L",
        (False, False, True, False, False): "Middle Finger",
        (True, True, False, False, False): "Gun 👈",
        (False, False, True, True, True): "Peace ✌",
        (True, False, False, True, False): "OK 👌",
        (False, True, False, True, False): "Scissors ✂️",
    }
    return gestures.get(tuple(states), "Unknown")

# Combine gesture from two hands
def classify_combined_gesture(gesture1, gesture2):
    if {"Rock 🤘", "Rock 🤘"} == {gesture1, gesture2}:
        return "Double Rock 🤘"
    elif {"Thumbs Up", "Thumbs Up"} == {gesture1, gesture2}:
        return "Victory 👍👍"
    elif {"Peace ✌", "Peace ✌"} == {gesture1, gesture2}:
        return "Double Peace ✌"
    return None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    left_gesture, right_gesture = None, None

    if result.multi_hand_landmarks and result.multi_handedness:
        for hand_landmarks, handedness in zip(result.multi_hand_landmarks, result.multi_handedness):
            label = handedness.classification[0].label  # 'Left' or 'Right'
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            states = get_finger_states(hand_landmarks, label)
            gesture = classify_gesture(states)

            if label == 'Left':
                left_gesture = gesture
                cv2.putText(frame, f'Left: {gesture}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            else:
                right_gesture = gesture
                cv2.putText(frame, f'Right: {gesture}', (frame.shape[1]-250, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Check for two-hand combo gesture
        if left_gesture and right_gesture:
            combo = classify_combined_gesture(left_gesture, right_gesture)
            if combo:
                cv2.putText(frame, f'Combo: {combo}', (int(frame.shape[1]/2)-100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3)

    cv2.imshow("Hand Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
