import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
import threading
import time
import queue

class AI_Gym_Trainer:
    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.cap = cv2.VideoCapture(0)
        
        # Audio Queue
        self.speech_queue = queue.Queue()
        self.stop_voice = False
        
        # Start Voice Thread
        threading.Thread(target=self._voice_worker, daemon=True).start()
        
        # Exercise State
        self.counter = 0 
        self.stage = None
        self.ready_position_confirmed = False
        self.ready_check_frames = 0
        self.last_speech_time = 0

    def _voice_worker(self):
        """
        ULTRA-ROBUST VOXY: Creates and kills the engine for EVERY sentence.
        This prevents the 'silent after first sentence' driver bug.
        """
        while not self.stop_voice:
            try:
                text = self.speech_queue.get(timeout=0.5)
                if text:
                    print(f"🔊 SENDING TO SPEAKERS: {text}")
                    # Initialize fresh inside the loop
                    engine = pyttsx3.init() 
                    engine.setProperty('rate', 150)
                    engine.say(text)
                    engine.runAndWait()
                    # Shutdown engine completely to free the driver
                    engine.stop()
                    del engine 
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Audio Driver Error: {e}")

    def speak(self, text, force=False):
        now = time.time()
        # Cooldown of 3 seconds unless it's a rep count
        if force or (now - self.last_speech_time > 3):
            self.last_speech_time = now
            self.speech_queue.put(text)

    def calculate_angle(self, a, b, c):
        a = np.array(a); b = np.array(b); c = np.array(c)
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians*180.0/np.pi)
        if angle > 180.0: angle = 360-angle
        return angle

    def run(self):
        # 1. INITIAL INSTRUCTION
        self.speak("System online. Step back to start.", force=True)
        
        with self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret: break
                
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(image)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                image = cv2.flip(image, 1)

                if results.pose_landmarks:
                    lm = results.pose_landmarks.landmark
                    
                    # COORDINATES (Left Arm)
                    s = [lm[11].x, lm[11].y]; e = [lm[13].x, lm[13].y]; w = [lm[15].x, lm[15].y]
                    angle = self.calculate_angle(s, e, w)

                    # STARTUP LOGIC: You must hold arm straight to 'activate' the trainer
                    if not self.ready_position_confirmed:
                        if angle > 160:
                            self.ready_check_frames += 1
                            cv2.putText(image, f"HOLDING: {self.ready_check_frames}/20", (10, 100), 1, 2, (0,255,0), 2)
                            if self.ready_check_frames == 5: self.speak("Good. Now hold it.")
                            if self.ready_check_frames > 20:
                                self.ready_position_confirmed = True
                                self.speak("Ready. Start curling now!", force=True)
                        else:
                            self.ready_check_frames = 0
                            cv2.putText(image, "STRAIGHTEN ARM", (10, 100), 1, 2, (0,0,255), 2)
                    
                    # COUNTING LOGIC
                    else:
                        if angle > 160: self.stage = "down"
                        if angle < 30 and self.stage == "down":
                            self.stage = "up"
                            self.counter += 1
                            # Force speak the number
                            self.speak(str(self.counter), force=True)

                    self.mp_drawing.draw_landmarks(image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

                cv2.putText(image, f"REPS: {self.counter}", (10, 50), 1, 3, (255,255,255), 3)
                cv2.imshow('Trainer', image)
                if cv2.waitKey(10) & 0xFF == ord('q'): break

        self.stop_voice = True
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = AI_Gym_Trainer()
    app.run()