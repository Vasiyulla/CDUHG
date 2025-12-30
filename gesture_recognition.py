import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

def recognize_gesture(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            gesture = analyze_hand_landmarks(hand_landmarks.landmark)
            return gesture
    return None

def analyze_hand_landmarks(landmarks):
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    middle_tip = landmarks[12]
    ring_tip = landmarks[16]
    pinky_tip = landmarks[20]

    if index_tip.y < thumb_tip.y:
        return 'list_files'
    elif index_tip.y < middle_tip.y < ring_tip.y < pinky_tip.y:
        return 'volume_up'
    elif index_tip.y > middle_tip.y > ring_tip.y > pinky_tip.y:
        return 'volume_down'
    elif thumb_tip.x < index_tip.x < middle_tip.x:
        return 'next_window'
    elif thumb_tip.x > index_tip.x > middle_tip.x:
        return 'previous_window'
    # Add more gesture recognition logic here
    return None

####################################################
# import mediapipe as mp
# import cv2
# mp_hands = mp.solutions.hands
# hands = mp_hands.Hands()
# mp_drawing = mp.solutions.drawing_utils

# def recognize_gesture(frame):
#     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     result = hands.process(rgb_frame)

#     if result.multi_hand_landmarks:
#         for hand_landmarks in result.multi_hand_landmarks:
#             mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
#             gesture = analyze_hand_landmarks(hand_landmarks.landmark)
#             return gesture
#     return None

# def analyze_hand_landmarks(landmarks):
#     # Placeholder for actual gesture recognition logic
#     thumb_tip = landmarks[4]
#     index_tip = landmarks[8]
    
#     if index_tip.y < thumb_tip.y:
#         return 'list_files'
#     # Add more gesture recognition logic here
#     return None
###########################################################
# import mediapipe as mp
# import cv2
# import cap
# mp_hands = mp.solutions.hands
# hands = mp_hands.Hands()
# mp_drawing = mp.solutions.drawing_utils

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break
    
#     # Convert the frame to RGB
#     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     result = hands.process(rgb_frame)
    
#     if result.multi_hand_landmarks:
#         for hand_landmarks in result.multi_hand_landmarks:
#             mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    
#     cv2.imshow('Hand Gesture Recognition', frame)
    
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()
