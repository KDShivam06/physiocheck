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
        
        # Set camera resolution to smaller size
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # State variables
        self.counter = 0 
        self.stage = None
        self.current_exercise = "curl"
        self.side = "left"
        
        # Voice feedback system with queue
        self.speech_queue = queue.Queue()
        self.voice_thread = None
        self.stop_voice_thread = False
        
        # Initialize voice thread
        print("✓ Initializing Voice Thread...")
        self.start_voice_thread()
        self.speech_queue.put(("Physio assistant ready.", True))
        
        self.last_feedback_time = 0
        self.feedback_cooldown = 2.5 
        
        # Body visibility tracking
        self.body_visible = False
        self.body_invisible_count = 0
        self.body_invisible_warning_given = False
        self.visibility_check_threshold = 15
        
        # Starting position validation
        self.ready_position_confirmed = False
        self.ready_position_check_frames = 0
        self.ready_position_required_frames = 10
        
        # Form tracking variables
        self.current_rep_valid = True
        self.rep_form_issues = []
        self.consecutive_bad_reps = 0
        self.last_corrective_feedback_time = 0
        
        # Pacing variables for "One... Two... Three..."
        self.pacing_state = 0 

        # Tracking for UI consistency
        self.last_hip_shoulder_angle = 0
        self.last_elbow_pos = 0
        self.elbow_stable_during_rep = True
        self.body_stable_during_rep = True
        self.gave_encouragement_at = set()

    def start_voice_thread(self):
        """Start a dedicated thread for voice feedback"""
        def voice_worker():
            while not self.stop_voice_thread:
                try:
                    item = self.speech_queue.get(timeout=0.5)
                    text = item[0] if isinstance(item, tuple) else item
                        
                    if text:
                        print(f"🔊 [SPEAKING]: {text}")
                        try:
                            engine = pyttsx3.init()
                            
                            # Try to set a female/calming voice
                            voices = engine.getProperty('voices')
                            for voice in voices:
                                if "female" in voice.name.lower() or "zira" in voice.name.lower():
                                    engine.setProperty('voice', voice.id)
                                    break
                            
                            engine.setProperty('rate', 145) # Slower, calming rate
                            engine.setProperty('volume', 1.0)
                            engine.say(text)
                            engine.runAndWait()
                            engine.stop()
                            del engine
                        except Exception as e:
                            print(f"✗ Speech error: {e}")      
                except queue.Empty:
                    continue
                except Exception as e:
                    print(f"✗ Voice worker error: {e}")
        
        self.voice_thread = threading.Thread(target=voice_worker, daemon=True)
        self.voice_thread.start()

    def speak_async(self, text, priority=False):
        """Non-blocking text-to-speech with queue"""
        current_time = time.time()
        
        if priority or (current_time - self.last_feedback_time >= self.feedback_cooldown):
            self.last_feedback_time = current_time
            if priority:
                # Clear queue for urgent correction
                while not self.speech_queue.empty():
                    try: self.speech_queue.get_nowait()
                    except: break
            try:
                self.speech_queue.put((text, priority), block=False)
            except queue.Full:
                pass

    def check_body_visibility(self, landmarks):
        """Check if key body parts are visible"""
        try:
            required = [11, 12, 23, 24] # Shoulders and Hips
            count = 0
            for idx in required:
                if landmarks[idx].visibility > 0.5: count += 1
            return count >= 3
        except: return False

    def reset_counter(self):
        self.counter = 0
        self.stage = None
        self.ready_position_confirmed = False
        self.current_rep_valid = True
        self.consecutive_bad_reps = 0
        self.pacing_state = 0
        
        # Human-centric intro instructions
        if self.current_exercise == "curl":
            self.speak_async("Bicep curl. Stand upright. Palms forward. Elbows close to your side.", priority=True)
        elif self.current_exercise == "squat":
            self.speak_async("Squats. Feet wider than hips. Toes out. Chest upright.", priority=True)
        elif self.current_exercise == "lift":
            self.speak_async("Lateral raises. Stand tall. Arms at sides. Palms facing inward.", priority=True)

    def calculate_angle(self, a, b, c):
        a = np.array(a) 
        b = np.array(b) 
        c = np.array(c) 
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians*180.0/np.pi)
        if angle > 180.0: angle = 360-angle
        return angle

    # =========================================================================
    # ORIGINAL UI CODE PRESERVED EXACTLY AS REQUESTED
    # =========================================================================
    def draw_status_box(self, image, angle, landmarks_coords, feedback_text="", body_angle=0):
        """Enhanced status box with real-time feedback (Original Layout)"""
        h, w = image.shape[:2]
        
        # Main status box
        box_color = (0, 255, 0) if self.body_visible else (0, 0, 255)
        cv2.rectangle(image, (0, 0), (600, 130), (30, 30, 30), -1)
        cv2.rectangle(image, (0, 0), (600, 130), box_color, 3)
        
        # REPS section (only valid reps)
        cv2.putText(image, 'VALID REPS', (20, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(image, str(self.counter), 
                    (20, 85), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 255, 0), 3, cv2.LINE_AA)
        
        # Bad reps counter
        if self.consecutive_bad_reps > 0:
            cv2.putText(image, f'Bad: {self.consecutive_bad_reps}', 
                        (20, 115), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 100, 255), 2, cv2.LINE_AA)
        
        # STAGE section
        cv2.putText(image, 'STAGE', (180, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        
        if self.current_exercise == "curl" and not self.ready_position_confirmed:
            stage_text = "READY?"
            stage_color = (255, 255, 0)
        else:
            stage_color = (0, 255, 255) if self.stage == "up" else (255, 165, 0)
            stage_text = str(self.stage).upper() if self.stage else "START"
        
        cv2.putText(image, stage_text, 
                    (180, 85), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, stage_color, 3, cv2.LINE_AA)
        
        # Form indicator
        if self.current_rep_valid:
            form_text = "✓ GOOD"
            form_color = (0, 255, 0)
        else:
            form_text = "✗ BAD"
            form_color = (0, 0, 255)
        
        cv2.putText(image, form_text, 
                    (180, 115), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, form_color, 2, cv2.LINE_AA)
        
        # EXERCISE section
        cv2.putText(image, 'EXERCISE', (380, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        display_text = f"{self.current_exercise.upper()} ({self.side[0].upper()})"
        cv2.putText(image, display_text, 
                    (380, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (100, 200, 255), 2, cv2.LINE_AA)
        cv2.putText(image, f"Arm:{int(angle)}° Body:{int(body_angle)}°", 
                    (380, 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        
        # Angle visualization on joint
        if self.body_visible and angle > 0:
            try:
                joint_pos = tuple(np.multiply(landmarks_coords, [w, h]).astype(int))
                
                # Draw large circle around joint
                cv2.circle(image, joint_pos, 15, (0, 255, 255), 3)
                cv2.circle(image, joint_pos, 10, (0, 255, 255), -1)
                
                # Draw angle text
                angle_text = f"{int(angle)}"
                cv2.putText(image, angle_text, 
                            (joint_pos[0] + 25, joint_pos[1] + 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3, cv2.LINE_AA)
            except Exception as e:
                pass
        
        # Feedback text box at bottom
        if feedback_text:
            feedback_height = 80
            if "✓" in feedback_text or "GOOD" in feedback_text or "PERFECT" in feedback_text:
                bg_color = (0, 150, 0)
            elif "✗" in feedback_text or "BAD" in feedback_text or "WRONG" in feedback_text or "⚠" in feedback_text:
                bg_color = (0, 0, 180)
            else:
                bg_color = (180, 100, 0)
            
            cv2.rectangle(image, (0, h - feedback_height), (w, h), bg_color, -1)
            cv2.rectangle(image, (0, h - feedback_height), (w, h), (255, 255, 255), 3)
            
            lines = feedback_text.split('\n')
            font_scale = 0.7
            thickness = 2
            line_height = 25
            
            for i, line in enumerate(lines[:2]):
                text_size = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
                text_x = max(10, (w - text_size[0]) // 2)
                text_y = h - feedback_height + 25 + (i * line_height)
                
                cv2.putText(image, line, (text_x, text_y), 
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    # =========================================================================
    # PHYSIOTHERAPIST LOGIC (Human Centric + Pacing)
    # =========================================================================
    def logic_bicep_curl(self, landmarks):
        # 1. Map landmarks
        if self.side == "left":
            s_idx, e_idx, w_idx, h_idx, ear_idx = 11, 13, 15, 23, 7 
        else:
            s_idx, e_idx, w_idx, h_idx, ear_idx = 12, 14, 16, 24, 8 

        shoulder = [landmarks[s_idx].x, landmarks[s_idx].y]
        elbow = [landmarks[e_idx].x, landmarks[e_idx].y]
        wrist = [landmarks[w_idx].x, landmarks[w_idx].y]
        hip = [landmarks[h_idx].x, landmarks[h_idx].y]
        ear = [landmarks[ear_idx].x, landmarks[ear_idx].y]

        angle = self.calculate_angle(shoulder, elbow, wrist)
        feedback_text = ""
        body_angle = 0 # Dummy for UI

        # 2. Ready Position Check
        if not self.ready_position_confirmed:
            if angle > 150:
                self.ready_position_check_frames += 1
                if self.ready_position_check_frames > 15:
                    self.ready_position_confirmed = True
                    self.speak_async("Begin. One...", priority=True)
                    self.stage = "up"
                    self.pacing_state = 1
            else:
                feedback_text = "Extend arm fully to start"
            return angle, elbow, feedback_text, body_angle

        # 3. Form Error Detection (Strict Geometric)
        # Elbow Drifting Forward
        if abs(elbow[0] - shoulder[0]) > 0.12: 
            feedback_text = "⚠ ELBOW DRIFT"
            self.speak_async("Bring your elbow back to your side.", priority=True)
            self.current_rep_valid = False
        
        # Shoulder Shrugging
        elif abs(shoulder[1] - ear[1]) < 0.10: 
            feedback_text = "⚠ SHRUGGING"
            self.speak_async("Relax your shoulders down.", priority=True)
            self.current_rep_valid = False

        # Leaning Backward
        elif abs(shoulder[0] - hip[0]) > 0.15:
            feedback_text = "⚠ LEANING"
            self.speak_async("Stand tall. Keep your torso still.", priority=True)
            self.current_rep_valid = False

        # 4. Movement & Pacing
        if self.stage == "up":
            # Pacing "One... Two... Three..."
            if angle < 120 and self.pacing_state == 1:
                self.speak_async("Two...")
                self.pacing_state = 2
            elif angle < 80 and self.pacing_state == 2:
                self.speak_async("Three...")
                self.pacing_state = 3
            
            # Top
            if angle < 35:
                self.stage = "peak"
                self.speak_async("Pause. Lower slowly.", priority=True)
                self.pacing_state = 0

        elif self.stage == "peak":
            if angle > 45:
                self.stage = "down"
                self.speak_async("One...")
                self.pacing_state = 1

        elif self.stage == "down":
            if angle > 80 and self.pacing_state == 1:
                self.speak_async("Two...")
                self.pacing_state = 2
            elif angle > 120 and self.pacing_state == 2:
                self.speak_async("Three...")
                self.pacing_state = 3
            
            # Rep Complete
            if angle > 160:
                if self.current_rep_valid:
                    self.counter += 1
                    feedback_text = "✓ GOOD REP"
                    self.speak_async(f"Repetition {self.counter}.", priority=True)
                else:
                    self.consecutive_bad_reps += 1
                    feedback_text = "✗ FIX FORM"
                
                self.stage = "up"
                self.pacing_state = 1
                self.current_rep_valid = True # Reset for next rep

        return angle, elbow, feedback_text, body_angle

    def logic_squats(self, landmarks):
        hip = [landmarks[23].x, landmarks[23].y]
        knee = [landmarks[25].x, landmarks[25].y]
        ankle = [landmarks[27].x, landmarks[27].y]
        shoulder = [landmarks[11].x, landmarks[11].y]
        
        # Valgus check
        r_knee = [landmarks[26].x, landmarks[26].y]
        r_hip = [landmarks[24].x, landmarks[24].y]

        angle = self.calculate_angle(hip, knee, ankle)
        feedback_text = ""
        body_angle = 0

        if not self.ready_position_confirmed:
            if angle > 165:
                self.ready_position_confirmed = True
                self.speak_async("Begin. Lower down slowly.", priority=True)
                self.stage = "down"
                self.pacing_state = 1
            else:
                feedback_text = "Stand straight to start"
            return angle, knee, feedback_text, body_angle

        # Error: Knees Caving In
        knee_dist = abs(knee[0] - r_knee[0])
        hip_dist = abs(hip[0] - r_hip[0])
        if self.stage == "down" and knee_dist < (hip_dist * 0.75):
            feedback_text = "⚠ KNEES INWARD"
            self.speak_async("Push knees outward.", priority=True)
            self.current_rep_valid = False

        # Error: Leaning Forward
        elif abs(shoulder[0] - hip[0]) > 0.3:
            feedback_text = "⚠ LEANING FORWARD"
            self.speak_async("Lift your chest.", priority=True)
            self.current_rep_valid = False

        if self.stage == "down":
            if angle < 160 and self.pacing_state == 1:
                self.speak_async("One...")
                self.pacing_state = 2
            elif angle < 140 and self.pacing_state == 2:
                self.speak_async("Two...")
                self.pacing_state = 3
            elif angle < 120 and self.pacing_state == 3:
                self.speak_async("Three...")
                self.pacing_state = 4
                
            if angle < 90:
                self.stage = "up"
                self.speak_async("Pause. Up.", priority=True)
                self.pacing_state = 1

        elif self.stage == "up":
            if angle > 100 and self.pacing_state == 1:
                self.speak_async("One...")
                self.pacing_state = 2
            elif angle > 140 and self.pacing_state == 2:
                self.speak_async("Two...")
                self.pacing_state = 3

            if angle > 165:
                if self.current_rep_valid:
                    self.counter += 1
                    feedback_text = "✓ GOOD SQUAT"
                    self.speak_async(f"Repetition {self.counter}.", priority=True)
                else:
                    self.consecutive_bad_reps += 1
                    feedback_text = "✗ FIX FORM"
                
                self.stage = "down"
                self.pacing_state = 1
                self.current_rep_valid = True

        return angle, knee, feedback_text, body_angle

    def logic_arm_lift(self, landmarks):
        shoulder = [landmarks[11].x, landmarks[11].y]
        elbow = [landmarks[13].x, landmarks[13].y]
        hip = [landmarks[23].x, landmarks[23].y]
        ear = [landmarks[7].x, landmarks[7].y]
        
        angle = self.calculate_angle(hip, shoulder, elbow)
        feedback_text = ""
        body_angle = 0

        if not self.ready_position_confirmed:
            if angle < 20:
                self.ready_position_confirmed = True
                self.speak_async("Begin. Raise arms out.", priority=True)
                self.stage = "up"
                self.pacing_state = 1
            else:
                feedback_text = "Arms at sides"
            return angle, shoulder, feedback_text, body_angle

        # Error: Too High
        if angle > 95:
            feedback_text = "⚠ TOO HIGH"
            self.speak_async("Stop at shoulder height.", priority=True)
            self.current_rep_valid = False
        # Error: Shrugging
        elif abs(shoulder[1] - ear[1]) < 0.10:
            feedback_text = "⚠ SHRUGGING"
            self.speak_async("Relax shoulders.", priority=True)
            self.current_rep_valid = False

        if self.stage == "up":
            if angle > 30 and self.pacing_state == 1:
                self.speak_async("Two...")
                self.pacing_state = 2
            elif angle > 60 and self.pacing_state == 2:
                self.speak_async("Three...")
                self.pacing_state = 3
            
            if angle > 80:
                self.stage = "down"
                self.speak_async("Pause. Lower.", priority=True)
                self.pacing_state = 1

        elif self.stage == "down":
            if angle < 60 and self.pacing_state == 1:
                self.speak_async("One...")
                self.pacing_state = 2
            elif angle < 30 and self.pacing_state == 2:
                self.speak_async("Two...")
                self.pacing_state = 3
            
            if angle < 15:
                if self.current_rep_valid:
                    self.counter += 1
                    feedback_text = "✓ GOOD REP"
                    self.speak_async(f"Repetition {self.counter}.", priority=True)
                else:
                    self.consecutive_bad_reps += 1
                    feedback_text = "✗ FIX FORM"
                
                self.stage = "up"
                self.pacing_state = 1
                self.current_rep_valid = True

        return angle, shoulder, feedback_text, body_angle

    def start_training(self):
        print("\n" + "="*60)
        print("Starting AI Gym Trainer...")
        print("="*60)
        
        self.speak_async("System ready.", priority=True)
        time.sleep(1)
        
        with self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret: break
                
                frame = cv2.flip(frame, 1)
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = pose.process(image)
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
                feedback_text = ""
                body_angle = 0
                
                # Check body visibility
                if results.pose_landmarks:
                    self.body_visible = self.check_body_visibility(results.pose_landmarks.landmark)
                    if not self.body_visible:
                        self.body_invisible_count += 1
                        if self.body_invisible_count > 30:
                             feedback_text = "Step back to show full body"
                    else:
                        self.body_invisible_count = 0
                
                try:
                    if self.body_visible and results.pose_landmarks:
                        landmarks = results.pose_landmarks.landmark
                        angle = 0
                        visualization_point = [0, 0]
                        
                        if self.current_exercise == "curl":
                            angle, visualization_point, fb_text, body_angle = self.logic_bicep_curl(landmarks)
                            if fb_text: feedback_text = fb_text
                        elif self.current_exercise == "squat":
                            angle, visualization_point, fb_text, body_angle = self.logic_squats(landmarks)
                            if fb_text: feedback_text = fb_text
                        elif self.current_exercise == "lift":
                            angle, visualization_point, fb_text, body_angle = self.logic_arm_lift(landmarks)
                            if fb_text: feedback_text = fb_text
                        
                        self.draw_status_box(image, angle, visualization_point, feedback_text, body_angle)
                    else:
                        self.draw_status_box(image, 0, [0.5, 0.5], feedback_text or "Position yourself", 0)
                        
                except Exception as e:
                    print(f"Error: {e}")
                
                if results.pose_landmarks:
                    self.mp_drawing.draw_landmarks(
                        image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                        self.mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2), 
                        self.mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                    )               
                
                # Instruction bar (Original Code Layout)
                h, w = image.shape[:2]
                instruction_y = h - 90
                cv2.rectangle(image, (0, instruction_y), (w, instruction_y + 30), (50, 50, 50), -1)
                cv2.putText(image, 'CONTROLS: [1] Curl  [2] Squat  [3] Lift  [L] Left  [R] Right  [Q] Quit', 
                            (10, instruction_y + 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

                cv2.imshow('AI Gym Trainer - Real-time Voice Feedback', image)

                k = cv2.waitKey(10) & 0xFF
                if k == ord('q'): 
                    self.speak_async("Session complete.", priority=True)
                    time.sleep(2)
                    break
                elif k == ord('1'): 
                    self.current_exercise = "curl"
                    self.reset_counter()
                elif k == ord('2'): 
                    self.current_exercise = "squat"
                    self.reset_counter()
                elif k == ord('3'): 
                    self.current_exercise = "lift"
                    self.reset_counter()
                elif k == ord('l'): 
                    self.side = "left"
                    self.reset_counter()
                elif k == ord('r'): 
                    self.side = "right"
                    self.reset_counter()

            self.stop_voice_thread = True
            self.cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    trainer = AI_Gym_Trainer()
    trainer.start_training()