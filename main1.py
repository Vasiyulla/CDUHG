import cv2
import mediapipe as mp
import pyautogui
import math
from enum import IntEnum
import screen_brightness_control as sbcontrol
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from google.protobuf.json_format import MessageToDict
import time

##
pyautogui.FAILSAFE=False
mp_drawing=mp.solutions.drawing_utils
mp_hands=mp.solutions.hands

####Entering the gesture
class Gest(IntEnum):
    #write fingure name into binary number
    FIST = 0
    PINKY = 1
    RING = 2
    MID = 4
    LAST3 = 7
    INDEX = 8
    FIRST2 = 12
    LAST4 = 15
    THUMB = 16    
    PALM = 31
    
    V_GEST = 33
    TWO_FINGER_CLOSED = 34
    PINCH_MAJOR = 35
    PINCH_MINOR = 36
    
    # NEW GESTURES
    THUMBS_UP = 42
    THUMBS_DOWN = 43
    ROCK = 44  # Rock & Roll gesture (pinky + index up)
    THREE_FINGERS = 45  # Index, middle, ring up
    FOUR_FINGERS = 46  # All except thumb
    PEACE_SIGN = 47  # V gesture with fingers apart
    OK_SIGN = 48  # Thumb and index forming circle
    POINTING_LEFT = 49
    POINTING_RIGHT = 50
    SWIPE_LEFT = 51
    SWIPE_RIGHT = 52
    SWIPE_UP = 53
    SWIPE_DOWN = 54
    
# Multi-handedness Labels
class HLabel(IntEnum):
    MINOR = 0
    MAJOR = 1

class HandRecog:
    """
    Convert Mediapipe Landmarks to recognizable Gestures.
    """
    
    def __init__(self, hand_label):
        self.finger = 0
        self.ori_gesture = Gest.PALM
        self.prev_gesture = Gest.PALM
        self.frame_count = 0
        self.hand_result = None
        self.hand_label = hand_label
        self.prev_hand_pos = None
        self.swipe_threshold = 0.12
        self.swipe_start_time = None
        self.swipe_start_pos = None
        self.gesture_buffer = []
        self.buffer_size = 3  # Reduced for faster response
    
    def update_hand_result(self, hand_result):
        self.hand_result = hand_result

    def get_signed_dist(self, point):
        """returns signed euclidean distance between 'point'."""
        sign = -1
        if self.hand_result.landmark[point[0]].y < self.hand_result.landmark[point[1]].y:
            sign = 1
        dist = (self.hand_result.landmark[point[0]].x - self.hand_result.landmark[point[1]].x)**2
        dist += (self.hand_result.landmark[point[0]].y - self.hand_result.landmark[point[1]].y)**2
        dist = math.sqrt(dist)
        return dist*sign
    
    def get_dist(self, point):
        """returns euclidean distance between 'point'."""
        dist = (self.hand_result.landmark[point[0]].x - self.hand_result.landmark[point[1]].x)**2
        dist += (self.hand_result.landmark[point[0]].y - self.hand_result.landmark[point[1]].y)**2
        dist = math.sqrt(dist)
        return dist
    
    def get_dz(self, point):
        """returns absolute difference on z-axis between 'point'."""
        return abs(self.hand_result.landmark[point[0]].z - self.hand_result.landmark[point[1]].z)
    
    def get_angle(self, p1, p2, p3):
        """Calculate angle between three points."""
        v1 = [self.hand_result.landmark[p1].x - self.hand_result.landmark[p2].x,
              self.hand_result.landmark[p1].y - self.hand_result.landmark[p2].y]
        v2 = [self.hand_result.landmark[p3].x - self.hand_result.landmark[p2].x,
              self.hand_result.landmark[p3].y - self.hand_result.landmark[p2].y]
        
        dot = v1[0]*v2[0] + v1[1]*v2[1]
        mag1 = math.sqrt(v1[0]**2 + v1[1]**2)
        mag2 = math.sqrt(v2[0]**2 + v2[1]**2)
        
        if mag1 == 0 or mag2 == 0:
            return 0
        
        cos_angle = dot / (mag1 * mag2)
        cos_angle = max(-1, min(1, cos_angle))
        return math.degrees(math.acos(cos_angle))
    
    def is_finger_extended(self, finger_tip, finger_pip, finger_mcp):
        """More accurate finger extension detection using angles."""
        angle = self.get_angle(finger_tip, finger_pip, finger_mcp)
        return angle > 140  # Finger is straight if angle > 140 degrees
    
    def is_thumb_extended(self):
        """Special thumb detection."""
        # Check thumb tip vs thumb base
        thumb_tip = self.hand_result.landmark[4]
        thumb_base = self.hand_result.landmark[2]
        index_mcp = self.hand_result.landmark[5]
        
        # Thumb is extended if tip is far from palm
        dist_to_palm = math.sqrt((thumb_tip.x - index_mcp.x)**2 + (thumb_tip.y - index_mcp.y)**2)
        return dist_to_palm > 0.1
    
    def set_finger_state(self):
        """set 'finger' by computing finger states more accurately."""
        if self.hand_result == None:
            return

        self.finger = 0
        
        # Thumb detection
        if self.is_thumb_extended():
            self.finger = self.finger | 16  # Thumb is up
        
        # Index finger (landmark 8, 6, 5)
        if self.is_finger_extended(8, 6, 5):
            self.finger = self.finger | 8
        
        # Middle finger (landmark 12, 10, 9)
        if self.is_finger_extended(12, 10, 9):
            self.finger = self.finger | 4
        
        # Ring finger (landmark 16, 14, 13)
        if self.is_finger_extended(16, 14, 13):
            self.finger = self.finger | 2
        
        # Pinky finger (landmark 20, 18, 17)
        if self.is_finger_extended(20, 18, 17):
            self.finger = self.finger | 1
    
    def detect_swipe(self):
        """Detect swipe gestures based on hand movement with time window."""
        if self.hand_result is None:
            return None
        
        current_time = time.time()
        current_pos = [self.hand_result.landmark[9].x, self.hand_result.landmark[9].y]
        
        # Initialize swipe tracking
        if self.swipe_start_pos is None:
            self.swipe_start_pos = current_pos
            self.swipe_start_time = current_time
            return None
        
        # Calculate movement
        dx = current_pos[0] - self.swipe_start_pos[0]
        dy = current_pos[1] - self.swipe_start_pos[1]
        time_elapsed = current_time - self.swipe_start_time
        
        # Reset if too much time has passed
        if time_elapsed > 0.5:
            self.swipe_start_pos = current_pos
            self.swipe_start_time = current_time
            return None
        
        swipe_gesture = None
        
        # Detect swipe if movement is significant and fast enough
        if time_elapsed > 0.1:  # Minimum time for swipe
            if abs(dx) > abs(dy) and abs(dx) > self.swipe_threshold:
                if dx > 0:
                    swipe_gesture = Gest.SWIPE_RIGHT
                else:
                    swipe_gesture = Gest.SWIPE_LEFT
                self.swipe_start_pos = current_pos
                self.swipe_start_time = current_time
                
            elif abs(dy) > abs(dx) and abs(dy) > self.swipe_threshold:
                if dy > 0:
                    swipe_gesture = Gest.SWIPE_DOWN
                else:
                    swipe_gesture = Gest.SWIPE_UP
                self.swipe_start_pos = current_pos
                self.swipe_start_time = current_time
        
        return swipe_gesture
    
    def get_gesture(self):
        """returns int representing gesture with improved accuracy and speed."""
        if self.hand_result == None:
            return Gest.PALM

        current_gesture = Gest.PALM
        
        # Priority 1: Pinch gestures (most specific)
        pinch_dist = self.get_dist([8, 4])
        if pinch_dist < 0.06:
            if self.finger in [Gest.LAST3, Gest.LAST4]:
                if self.hand_label == HLabel.MINOR:
                    current_gesture = Gest.PINCH_MINOR
                else:
                    current_gesture = Gest.PINCH_MAJOR
                # Immediate recognition for pinch
                self.ori_gesture = current_gesture
                return current_gesture
        
        # Priority 2: OK sign (thumb + index circle)
        if pinch_dist < 0.06 and (self.finger & Gest.LAST3) == Gest.LAST3:
            current_gesture = Gest.OK_SIGN
            self.ori_gesture = current_gesture
            return current_gesture
        
        # Priority 3: Specific multi-finger gestures
        # Thumbs up (only thumb extended, all others closed)
        if self.finger == Gest.THUMB:
            thumb_tip = self.hand_result.landmark[4]
            thumb_base = self.hand_result.landmark[2]
            if thumb_tip.y < thumb_base.y - 0.05:
                current_gesture = Gest.THUMBS_UP
            else:
                current_gesture = Gest.THUMBS_DOWN
            self.ori_gesture = current_gesture
            return current_gesture
        
        # Rock gesture (pinky and index up, others down)
        elif self.finger == (Gest.INDEX | Gest.PINKY):
            current_gesture = Gest.ROCK
            self.ori_gesture = current_gesture
            return current_gesture
        
        # Peace sign / V gesture (index and middle up)
        elif self.finger == Gest.FIRST2:
            point = [[8,12],[5,9]]
            dist1 = self.get_dist(point[0])
            dist2 = self.get_dist(point[1])
            ratio = dist1/dist2
            if ratio > 1.5:
                current_gesture = Gest.V_GEST
            else:
                if self.get_dz([8,12]) < 0.1:
                    current_gesture = Gest.TWO_FINGER_CLOSED
                else:
                    current_gesture = Gest.MID
            self.ori_gesture = current_gesture
            return current_gesture
        
        # Three fingers up (index, middle, ring)
        elif self.finger == (Gest.INDEX | Gest.MID | Gest.RING):
            current_gesture = Gest.THREE_FINGERS
            self.ori_gesture = current_gesture
            return current_gesture
        
        # Four fingers up (all except thumb)
        elif self.finger == Gest.LAST4:
            current_gesture = Gest.FOUR_FINGERS
            self.ori_gesture = current_gesture
            return current_gesture
        
        # Priority 4: Single finger gestures
        # Pointing with index finger
        elif self.finger == Gest.INDEX:
            index_tip = self.hand_result.landmark[8]
            wrist = self.hand_result.landmark[0]
            
            # Check horizontal direction
            if abs(index_tip.x - wrist.x) > 0.2:
                if index_tip.x > wrist.x:
                    current_gesture = Gest.POINTING_RIGHT
                else:
                    current_gesture = Gest.POINTING_LEFT
            else:
                current_gesture = Gest.INDEX
            
            self.ori_gesture = current_gesture
            return current_gesture
        
        # Fist
        elif self.finger == Gest.FIST:
            current_gesture = Gest.FIST
            self.ori_gesture = current_gesture
            return current_gesture
        
        # Middle finger alone
        elif self.finger == Gest.MID:
            current_gesture = Gest.MID
            self.ori_gesture = current_gesture
            return current_gesture
        
        # Priority 5: Open palm - check for swipes
        elif self.finger == Gest.PALM:
            swipe = self.detect_swipe()
            if swipe is not None:
                current_gesture = swipe
                # Immediate swipe recognition
                self.ori_gesture = current_gesture
                return current_gesture
            else:
                current_gesture = Gest.PALM
        
        else:
            current_gesture = self.finger
        
        # Use gesture buffer for stability on remaining gestures
        self.gesture_buffer.append(current_gesture)
        if len(self.gesture_buffer) > self.buffer_size:
            self.gesture_buffer.pop(0)
        
        # If majority of buffer agrees, update gesture
        if len(self.gesture_buffer) == self.buffer_size:
            if self.gesture_buffer.count(current_gesture) >= 2:
                self.ori_gesture = current_gesture
        
        return self.ori_gesture


class Controller:
    """Executes commands according to detected gestures."""

    tx_old = 0
    ty_old = 0
    trial = True
    flag = False
    grabflag = False
    pinchmajorflag = False
    pinchminorflag = False
    pinchstartxcoord = None
    pinchstartycoord = None
    pinchdirectionflag = None
    prevpinchlv = 0
    pinchlv = 0
    framecount = 0
    prev_hand = None
    pinch_threshold = 0.3
    last_gesture_time = {}
    gesture_cooldown = 0.5  # Cooldown in seconds
    
    @staticmethod
    def can_execute_gesture(gesture):
        """Check if enough time has passed since last execution."""
        current_time = time.time()
        if gesture not in Controller.last_gesture_time:
            Controller.last_gesture_time[gesture] = current_time
            return True
        
        if current_time - Controller.last_gesture_time[gesture] > Controller.gesture_cooldown:
            Controller.last_gesture_time[gesture] = current_time
            return True
        
        return False
    
    def getpinchylv(hand_result):
        """returns distance beween starting pinch y coord and current hand position y coord."""
        dist = round((Controller.pinchstartycoord - hand_result.landmark[8].y)*10,1)
        return dist

    def getpinchxlv(hand_result):
        """returns distance beween starting pinch x coord and current hand position x coord."""
        dist = round((hand_result.landmark[8].x - Controller.pinchstartxcoord)*10,1)
        return dist
    
    def changesystembrightness():
        """sets system brightness based on 'Controller.pinchlv'."""
        try:
            currentBrightnessLv = sbcontrol.get_brightness(display=0)/100.0
            currentBrightnessLv += Controller.pinchlv/50.0
            if currentBrightnessLv > 1.0:
                currentBrightnessLv = 1.0
            elif currentBrightnessLv < 0.0:
                currentBrightnessLv = 0.0       
            sbcontrol.fade_brightness(int(100*currentBrightnessLv), start=sbcontrol.get_brightness(display=0))
        except:
            pass
    
    def changesystemvolume():
        """sets system volume based on 'Controller.pinchlv'."""
        try:
            devices = AudioUtilities.GetSpeakers()
            interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
            volume = cast(interface, POINTER(IAudioEndpointVolume))
            currentVolumeLv = volume.GetMasterVolumeLevelScalar()
            currentVolumeLv += Controller.pinchlv/50.0
            if currentVolumeLv > 1.0:
                currentVolumeLv = 1.0
            elif currentVolumeLv < 0.0:
                currentVolumeLv = 0.0
            volume.SetMasterVolumeLevelScalar(currentVolumeLv, None)
        except:
            pass
    
    def scrollVertical():
        """scrolls on screen vertically."""
        pyautogui.scroll(120 if Controller.pinchlv>0.0 else -120)
    
    def scrollHorizontal():
        """scrolls on screen horizontally."""
        pyautogui.keyDown('shift')
        pyautogui.keyDown('ctrl')
        pyautogui.scroll(-120 if Controller.pinchlv>0.0 else 120)
        pyautogui.keyUp('ctrl')
        pyautogui.keyUp('shift')

    def get_position(hand_result):
        """returns coordinates of current hand position."""
        point = 9
        position = [hand_result.landmark[point].x, hand_result.landmark[point].y]
        sx, sy = pyautogui.size()
        x_old, y_old = pyautogui.position()
        x = int(position[0]*sx)
        y = int(position[1]*sy)
        if Controller.prev_hand is None:
            Controller.prev_hand = x, y
        delta_x = x - Controller.prev_hand[0]
        delta_y = y - Controller.prev_hand[1]

        distsq = delta_x**2 + delta_y**2
        ratio = 1
        Controller.prev_hand = [x, y]

        if distsq <= 25:
            ratio = 0
        elif distsq <= 900:
            ratio = 0.07 * (distsq ** (1/2))
        else:
            ratio = 2.1
        x, y = x_old + delta_x*ratio, y_old + delta_y*ratio
        return (x, y)

    def pinch_control_init(hand_result):
        """Initializes attributes for pinch gesture."""
        Controller.pinchstartxcoord = hand_result.landmark[8].x
        Controller.pinchstartycoord = hand_result.landmark[8].y
        Controller.pinchlv = 0
        Controller.prevpinchlv = 0
        Controller.framecount = 0

    def pinch_control(hand_result, controlHorizontal, controlVertical):
        """calls 'controlHorizontal' or 'controlVertical' based on pinch flags."""
        if Controller.framecount == 3:  # Reduced from 5 for faster response
            Controller.framecount = 0
            Controller.pinchlv = Controller.prevpinchlv

            if Controller.pinchdirectionflag == True:
                controlHorizontal()
            elif Controller.pinchdirectionflag == False:
                controlVertical()

        lvx = Controller.getpinchxlv(hand_result)
        lvy = Controller.getpinchylv(hand_result)
            
        if abs(lvy) > abs(lvx) and abs(lvy) > Controller.pinch_threshold:
            Controller.pinchdirectionflag = False
            if abs(Controller.prevpinchlv - lvy) < Controller.pinch_threshold:
                Controller.framecount += 1
            else:
                Controller.prevpinchlv = lvy
                Controller.framecount = 0

        elif abs(lvx) > Controller.pinch_threshold:
            Controller.pinchdirectionflag = True
            if abs(Controller.prevpinchlv - lvx) < Controller.pinch_threshold:
                Controller.framecount += 1
            else:
                Controller.prevpinchlv = lvx
                Controller.framecount = 0

    def handle_controls(gesture, hand_result):  
        """Implements all gesture functionality with immediate response."""      
        x, y = None, None
        if gesture != Gest.PALM:
            x, y = Controller.get_position(hand_result)
        
        # Flag reset
        if gesture != Gest.FIST and Controller.grabflag:
            Controller.grabflag = False
            pyautogui.mouseUp(button="left")

        if gesture != Gest.PINCH_MAJOR and Controller.pinchmajorflag:
            Controller.pinchmajorflag = False

        if gesture != Gest.PINCH_MINOR and Controller.pinchminorflag:
            Controller.pinchminorflag = False

        # Continuous gestures (no cooldown needed)
        if gesture == Gest.V_GEST:
            Controller.flag = True
            pyautogui.moveTo(x, y, duration=0.1)

        elif gesture == Gest.FIST:
            if not Controller.grabflag: 
                Controller.grabflag = True
                pyautogui.mouseDown(button="left")
            pyautogui.moveTo(x, y, duration=0.1)

        elif gesture == Gest.MID and Controller.flag:
            if Controller.can_execute_gesture(Gest.MID):
                pyautogui.click()
                Controller.flag = False

        elif gesture == Gest.INDEX and Controller.flag:
            if Controller.can_execute_gesture(Gest.INDEX):
                pyautogui.click(button='right')
                Controller.flag = False

        elif gesture == Gest.TWO_FINGER_CLOSED and Controller.flag:
            if Controller.can_execute_gesture(Gest.TWO_FINGER_CLOSED):
                pyautogui.doubleClick()
                Controller.flag = False

        elif gesture == Gest.PINCH_MINOR:
            if Controller.pinchminorflag == False:
                Controller.pinch_control_init(hand_result)
                Controller.pinchminorflag = True
            Controller.pinch_control(hand_result, Controller.scrollHorizontal, Controller.scrollVertical)
        
        elif gesture == Gest.PINCH_MAJOR:
            if Controller.pinchmajorflag == False:
                Controller.pinch_control_init(hand_result)
                Controller.pinchmajorflag = True
            Controller.pinch_control(hand_result, Controller.changesystembrightness, Controller.changesystemvolume)
        
        # NEW GESTURE IMPLEMENTATIONS with cooldown
        elif gesture == Gest.THUMBS_UP:
            if Controller.can_execute_gesture(Gest.THUMBS_UP):
                pyautogui.press('playpause')
        
        elif gesture == Gest.THUMBS_DOWN:
            if Controller.can_execute_gesture(Gest.THUMBS_DOWN):
                pyautogui.press('stop')
        
        elif gesture == Gest.ROCK:
            if Controller.can_execute_gesture(Gest.ROCK):
                pyautogui.press('volumeup')
        
        elif gesture == Gest.THREE_FINGERS:
            if Controller.can_execute_gesture(Gest.THREE_FINGERS):
                pyautogui.hotkey('win', 'shift', 's')
        
        elif gesture == Gest.FOUR_FINGERS:
            if Controller.can_execute_gesture(Gest.FOUR_FINGERS):
                pyautogui.hotkey('win', 'd')
        
        elif gesture == Gest.OK_SIGN:
            if Controller.can_execute_gesture(Gest.OK_SIGN):
                pyautogui.hotkey('win', 'down')
        
        elif gesture == Gest.SWIPE_LEFT:
            if Controller.can_execute_gesture(Gest.SWIPE_LEFT):
                pyautogui.press('prevtrack')
        
        elif gesture == Gest.SWIPE_RIGHT:
            if Controller.can_execute_gesture(Gest.SWIPE_RIGHT):
                pyautogui.press('nexttrack')
        
        elif gesture == Gest.SWIPE_UP:
            if Controller.can_execute_gesture(Gest.SWIPE_UP):
                pyautogui.press('volumeup')
        
        elif gesture == Gest.SWIPE_DOWN:
            if Controller.can_execute_gesture(Gest.SWIPE_DOWN):
                pyautogui.press('volumedown')
        
        elif gesture == Gest.POINTING_LEFT:
            if Controller.can_execute_gesture(Gest.POINTING_LEFT):
                pyautogui.hotkey('alt', 'left')
        
        elif gesture == Gest.POINTING_RIGHT:
            if Controller.can_execute_gesture(Gest.POINTING_RIGHT):
                pyautogui.hotkey('alt', 'right')


class GestureController:
    """Handles camera, obtain landmarks from mediapipe, entry point for whole program."""
    
    gc_mode = 0
    cap = None
    CAM_HEIGHT = None
    CAM_WIDTH = None
    hr_major = None
    hr_minor = None
    dom_hand = True

    def __init__(self):
        """Initializes attributes."""
        GestureController.gc_mode = 1
        GestureController.cap = cv2.VideoCapture(0)
        # Increase camera FPS for better responsiveness
        GestureController.cap.set(cv2.CAP_PROP_FPS, 60)
        GestureController.CAM_HEIGHT = GestureController.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        GestureController.CAM_WIDTH = GestureController.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    
    def classify_hands(results):
        """sets 'hr_major', 'hr_minor' based on classification(left, right) of hand."""
        left, right = None, None
        try:
            handedness_dict = MessageToDict(results.multi_handedness[0])
            if handedness_dict['classification'][0]['label'] == 'Right':
                right = results.multi_hand_landmarks[0]
            else:
                left = results.multi_hand_landmarks[0]
        except:
            pass

        try:
            handedness_dict = MessageToDict(results.multi_handedness[1])
            if handedness_dict['classification'][0]['label'] == 'Right':
                right = results.multi_hand_landmarks[1]
            else:
                left = results.multi_hand_landmarks[1]
        except:
            pass
        
        if GestureController.dom_hand == True:
            GestureController.hr_major = right
            GestureController.hr_minor = left
        else:
            GestureController.hr_major = left
            GestureController.hr_minor = right

    def start(self):
        """Entry point of whole program."""
        
        handmajor = HandRecog(HLabel.MAJOR)
        handminor = HandRecog(HLabel.MINOR)

        # Reduced min_detection_confidence for faster initial detection
        with mp_hands.Hands(
            max_num_hands=2, 
            min_detection_confidence=0.7,  # Increased for accuracy
            min_tracking_confidence=0.7,   # Increased for accuracy
            model_complexity=1  # Use higher complexity model
        ) as hands:
            while GestureController.cap.isOpened() and GestureController.gc_mode:
                success, image = GestureController.cap.read()

                if not success:
                    print("Ignoring empty camera frame.")
                    continue
                
                image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = hands.process(image)
                
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                if results.multi_hand_landmarks:                   
                    GestureController.classify_hands(results)
                    handmajor.update_hand_result(GestureController.hr_major)
                    handminor.update_hand_result(GestureController.hr_minor)

                    handmajor.set_finger_state()
                    handminor.set_finger_state()
                    gest_name = handminor.get_gesture()

                    if gest_name == Gest.PINCH_MINOR:
                        Controller.handle_controls(gest_name, handminor.hand_result)
                    else:
                        gest_name = handmajor.get_gesture()
                        Controller.handle_controls(gest_name, handmajor.hand_result)
                    
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    
                    # Display current gesture on screen with color coding
                    gesture_text = f"Gesture: {gest_name.name if hasattr(gest_name, 'name') else str(gest_name)}"
                    # Green for recognized gestures, yellow for palm
                    color = (0, 255, 0) if gest_name != Gest.PALM else (0, 255, 255)
                    cv2.putText(image, gesture_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                    
                    # Display finger state for debugging
                    finger_text = f"Fingers: {bin(handmajor.finger)}"
                    cv2.putText(image, finger_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                else:
                    Controller.prev_hand = None
                    cv2.putText(image, "No hand detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    
                cv2.imshow('Gesture Controller', image)
                if cv2.waitKey(1) & 0xFF == 13:  # Reduced from 5 to 1 for faster response
                    break
                    
        GestureController.cap.release()
        cv2.destroyAllWindows()


# Uncomment to run directly
if __name__ == "__main__":
    print("=== Gesture Controller ===")
    print("Press Enter to exit")
    print("\nGestures:")
    print("- V-Shape: Move cursor")
    print("- Fist: Drag (hold left click)")
    print("- Middle finger (after V): Left click")
    print("- Index finger (after V): Right click")
    print("- Two fingers closed (after V): Double click")
    print("- Thumbs Up: Play/Pause")
    print("- Thumbs Down: Stop")
    print("- Rock Sign: Volume Up")
    print("- Three Fingers: Screenshot")
    print("- Four Fingers: Show Desktop")
    print("- OK Sign: Minimize Window")
    print("- Point Left/Right: Browser Navigation")
    print("- Swipe Left/Right: Previous/Next Track")
    print("- Swipe Up/Down: Volume Up/Down")
    print("- Pinch (Major hand): Brightness/Volume control")
    print("- Pinch (Minor hand): Scroll")
    print("\n")
    
    gc1 = GestureController()
    gc1.start()