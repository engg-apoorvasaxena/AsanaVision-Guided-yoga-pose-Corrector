"""
PROFESSIONAL YOGA POSE DETECTION APP
Real-time feedback with joint-specific instructions
"""

import cv2
import mediapipe as mp
import numpy as np
import time

# Initialize MediaPipe
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

class YogaPoseDetector:
    def __init__(self):
        # Reference poses with target angles
        self.reference_poses = {
            'Warrior II': [175, 175, 90, 90, 175, 175, 100, 175],
            'T Pose': [175, 175, 90, 90, 180, 180, 180, 180],
            'Tree Pose': [175, 175, 180, 180, 175, 175, 180, 40],
            'Mountain Pose': [180, 180, 180, 180, 180, 180, 180, 180],
            'Chair Pose': [175, 175, 45, 45, 100, 100, 90, 90]
        }
        
        # Pose descriptions
        self.pose_descriptions = {
            'Warrior II': 'Strong warrior stance with arms extended',
            'T Pose': 'Stand straight, arms horizontally extended',
            'Tree Pose': 'Balance on one leg, foot on inner thigh',
            'Mountain Pose': 'Stand tall, feet together, arms at sides',
            'Chair Pose': 'Squat position with arms raised overhead'
        }
        
        # Tracking variables
        self.current_pose = 'Warrior II'
        self.pose_start_time = None
        self.pose_hold_time = 0
        self.is_holding_pose = False
        self.hold_threshold = 0.8
        self.total_session_time = 0
        self.pose_count = 0
        self.show_pose_menu = True
        self.best_accuracy = 0
        
    def calculate_angle(self, a, b, c):
        """Calculate angle between three points"""
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        
        if angle > 180.0:
            angle = 360 - angle
        return angle
    
    def extract_angles(self, landmarks):
        """Extract 8 key angles from pose landmarks"""
        rs = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
              landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        re = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
              landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
        rw = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
              landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
        ls = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
              landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        le = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
              landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
        lw = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
              landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
        rh = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
              landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
        lh = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
              landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        rk = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
              landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
        lk = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
              landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
        ra = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
              landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
        la = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
              landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
        
        angles = [
            int(self.calculate_angle(rs, re, rw)),  # Right elbow
            int(self.calculate_angle(ls, le, lw)),  # Left elbow
            int(self.calculate_angle(re, rs, rh)),  # Right shoulder
            int(self.calculate_angle(le, ls, lh)),  # Left shoulder
            int(self.calculate_angle(rs, rh, rk)),  # Right hip
            int(self.calculate_angle(ls, lh, lk)),  # Left hip
            int(self.calculate_angle(rh, rk, ra)),  # Right knee
            int(self.calculate_angle(lh, lk, la))   # Left knee
        ]
        
        angle_points = [re, le, rs, ls, rh, lh, rk, lk]
        return angles, angle_points
    
    def calculate_accuracy(self, user_angles, target_angles):
        """Calculate pose accuracy as percentage"""
        differences = []
        for ua, ta in zip(user_angles, target_angles):
            if ta == 0:
                ta = 1
            diff = abs(ua - ta) / ta
            differences.append(diff)
        
        avg_diff = sum(differences) / len(differences)
        accuracy = max(0, 1 - avg_diff)
        return accuracy
    
    def get_detailed_feedback(self, user_angles, target_angles, angle_points):
        """Generate detailed feedback for pose correction"""
        feedback_data = []
        
        instructions_bend = [
            'Bend arm',
            'Bend arm',
            'Lower arm',
            'Lower arm',
            'Bend forward',
            'Bend forward',
            'Bend knee',
            'Bend knee'
        ]
        
        instructions_extend = [
            'Straighten arm',
            'Straighten arm',
            'Raise arm',
            'Raise arm',
            'Straighten hip',
            'Straighten hip',
            'Straighten leg',
            'Straighten leg'
        ]
        
        for i, (ua, ta, point) in enumerate(zip(user_angles, target_angles, angle_points)):
            diff = ua - ta
            if diff < -15:
                feedback_data.append({
                    'position': point,
                    'instruction': instructions_extend[i],
                    'index': i
                })
            elif diff > 15:
                feedback_data.append({
                    'position': point,
                    'instruction': instructions_bend[i],
                    'index': i
                })
        
        return feedback_data
    
    def draw_pose_selection_menu(self, image):
        """Draw modern pose selection menu"""
        h, w, _ = image.shape
        
        # Dark overlay with gradient effect
        overlay = image.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.92, image, 0.08, 0, image)
        
        # Modern title
        title = "Select Your Yoga Pose"
        cv2.putText(image, title, (w//2 - 200, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.3, (100, 200, 255), 3)
        
        # Subtitle
        cv2.putText(image, "Choose a pose to begin your practice", (w//2 - 180, 110),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
        
        # Pose cards
        start_y = 160
        card_height = 70
        card_width = 600
        spacing = 15
        
        for i, (pose_name, description) in enumerate(self.pose_descriptions.items()):
            y = start_y + i * (card_height + spacing)
            x = w//2 - card_width//2
            
            # Card background with gradient
            if pose_name == self.current_pose:
                color1 = (80, 180, 100)
                color2 = (60, 150, 80)
                border_color = (100, 220, 120)
            else:
                color1 = (50, 50, 50)
                color2 = (40, 40, 40)
                border_color = (80, 80, 80)
            
            # Draw card with rounded effect
            cv2.rectangle(image, (x, y), (x + card_width, y + card_height), color1, -1)
            cv2.rectangle(image, (x, y), (x + card_width, y + card_height), border_color, 2)
            
            # Number badge
            badge_size = 40
            cv2.circle(image, (x + 35, y + card_height//2), badge_size//2, (100, 200, 255), -1)
            cv2.putText(image, str(i+1), (x + 26, y + card_height//2 + 8),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Pose name
            cv2.putText(image, pose_name, (x + 80, y + 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            
            # Description
            cv2.putText(image, description, (x + 80, y + 55),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        
        # Bottom instructions with icons
        inst_y = h - 60
        cv2.rectangle(image, (0, inst_y - 20), (w, h), (30, 30, 30), -1)
        cv2.putText(image, "Press [1-5] to select  |  [SPACE] to start  |  [Q] to quit",
                   (w//2 - 330, inst_y + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    
    def draw_top_bar(self, image, accuracy):
        """Draw professional top bar"""
        h, w, _ = image.shape
        
        # Top bar background
        overlay = image.copy()
        cv2.rectangle(overlay, (0, 0), (w, 80), (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.85, image, 0.15, 0, image)
        
        # Pose name with icon
        cv2.putText(image, "Current:", (20, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        cv2.putText(image, self.current_pose, (20, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (100, 200, 255), 2)
        
        # Accuracy meter in center
        meter_x = w//2 - 150
        meter_y = 25
        meter_width = 300
        meter_height = 30
        
        # Background
        cv2.rectangle(image, (meter_x, meter_y), (meter_x + meter_width, meter_y + meter_height),
                     (60, 60, 60), -1)
        
        # Fill based on accuracy
        fill_width = int(meter_width * accuracy)
        if accuracy > 0.8:
            color = (80, 200, 100)
        elif accuracy > 0.6:
            color = (100, 180, 255)
        else:
            color = (100, 100, 255)
        
        cv2.rectangle(image, (meter_x, meter_y), (meter_x + fill_width, meter_y + meter_height),
                     color, -1)
        
        # Percentage text
        cv2.putText(image, f"{int(accuracy * 100)}%", (meter_x + meter_width//2 - 30, meter_y + 22),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Timer on right
        minutes = int(self.pose_hold_time // 60)
        seconds = int(self.pose_hold_time % 60)
        timer_text = f"{minutes:02d}:{seconds:02d}"
        
        cv2.putText(image, "Hold Time:", (w - 200, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        
        timer_color = (80, 200, 100) if self.is_holding_pose else (100, 100, 100)
        cv2.putText(image, timer_text, (w - 200, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, timer_color, 2)
    
    def draw_status_badge(self, image, accuracy):
        """Draw status badge"""
        h, w, _ = image.shape
        
        if accuracy >= 0.8:
            status_text = "PERFECT!"
            color = (80, 200, 100)
        elif accuracy >= 0.6:
            status_text = "GOOD"
            color = (100, 180, 255)
        else:
            status_text = "ADJUST"
            color = (100, 100, 255)
        
        # Badge position (top right corner)
        badge_x = w - 180
        badge_y = 100
        badge_w = 160
        badge_h = 50
        
        overlay = image.copy()
        cv2.rectangle(overlay, (badge_x, badge_y), (badge_x + badge_w, badge_y + badge_h),
                     color, -1)
        cv2.addWeighted(overlay, 0.8, image, 0.2, 0, image)
        
        cv2.rectangle(image, (badge_x, badge_y), (badge_x + badge_w, badge_y + badge_h),
                     (255, 255, 255), 2)
        
        text_size = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        text_x = badge_x + (badge_w - text_size[0]) // 2
        text_y = badge_y + (badge_h + text_size[1]) // 2
        
        cv2.putText(image, status_text, (text_x, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    def draw_stats_panel(self, image):
        """Draw compact stats panel"""
        h, w, _ = image.shape
        
        # Bottom panel
        panel_h = 60
        overlay = image.copy()
        cv2.rectangle(overlay, (0, h - panel_h), (w, h), (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.85, image, 0.15, 0, image)
        
        y = h - 35
        
        # Session time
        cv2.putText(image, f"Session: {int(self.total_session_time)}s", (20, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # Poses held
        cv2.putText(image, f"Poses Held: {self.pose_count}", (w//2 - 80, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # Best accuracy
        cv2.putText(image, f"Best: {int(self.best_accuracy * 100)}%", (w - 150, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # Controls hint
        cv2.putText(image, "[SPACE] Menu | [R] Reset | [Q] Quit", (w//2 - 180, h - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1)
    
    def draw_joint_instructions(self, image, feedback_data):
        """Draw instructions directly on joints"""
        h, w, _ = image.shape
        
        for data in feedback_data:
            x = int(data['position'][0] * w)
            y = int(data['position'][1] * h)
            instruction = data['instruction']
            
            # Draw attention circle (pulsing effect would be cool but static for now)
            cv2.circle(image, (x, y), 25, (0, 0, 255), 3)
            cv2.circle(image, (x, y), 30, (255, 255, 255), 1)
            
            # Text background
            text = instruction
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            
            # Position text above or below joint based on y position
            if y < h // 2:
                text_y = y + 50
            else:
                text_y = y - 30
            
            text_x = x - text_size[0] // 2
            
            # Ensure text stays within bounds
            text_x = max(10, min(text_x, w - text_size[0] - 10))
            
            # Draw text background
            padding = 5
            cv2.rectangle(image, 
                         (text_x - padding, text_y - text_size[1] - padding),
                         (text_x + text_size[0] + padding, text_y + padding),
                         (0, 0, 0), -1)
            
            cv2.rectangle(image, 
                         (text_x - padding, text_y - text_size[1] - padding),
                         (text_x + text_size[0] + padding, text_y + padding),
                         (0, 0, 255), 2)
            
            # Draw text
            cv2.putText(image, text, (text_x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    def run(self):
        """Main application loop"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Error: Cannot access camera")
            return
        
        print("\n🧘 PROFESSIONAL YOGA POSE DETECTION")
        print("="*50)
        print("Controls:")
        print("  1-5: Select pose")
        print("  SPACE: Toggle menu")
        print("  F: Toggle fullscreen")
        print("  R: Reset timer")
        print("  Q: Quit")
        print("="*50 + "\n")
        
        # Create named window with ability to resize
        window_name = 'Yoga Pose Detector - Professional Edition'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1280, 720)
        
        is_fullscreen = False
        session_start = time.time()
        
        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame = cv2.flip(frame, 1)
                
                # Process image
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = pose.process(image)
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
                # Resize to standard HD
                image = cv2.resize(image, (1280, 720))
                
                if self.show_pose_menu:
                    self.draw_pose_selection_menu(image)
                else:
                    try:
                        if results.pose_landmarks:
                            landmarks = results.pose_landmarks.landmark
                            
                            # Extract angles
                            user_angles, angle_points = self.extract_angles(landmarks)
                            target_angles = self.reference_poses[self.current_pose]
                            
                            # Calculate accuracy
                            accuracy = self.calculate_accuracy(user_angles, target_angles)
                            
                            # Update best accuracy
                            if accuracy > self.best_accuracy:
                                self.best_accuracy = accuracy
                            
                            # Get feedback
                            feedback_data = self.get_detailed_feedback(
                                user_angles, target_angles, angle_points)
                            
                            # Track pose holding
                            if accuracy >= self.hold_threshold:
                                if not self.is_holding_pose:
                                    self.is_holding_pose = True
                                    self.pose_start_time = time.time()
                                    self.pose_count += 1
                                self.pose_hold_time = time.time() - self.pose_start_time
                            else:
                                self.is_holding_pose = False
                                self.pose_hold_time = 0
                            
                            # Draw skeleton with custom colors
                            mp_drawing.draw_landmarks(
                                image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(255, 100, 100), thickness=2, circle_radius=4),
                                mp_drawing.DrawingSpec(color=(100, 255, 100), thickness=3, circle_radius=2))
                            
                            # Draw all UI elements
                            self.draw_top_bar(image, accuracy)
                            self.draw_status_badge(image, accuracy)
                            self.draw_joint_instructions(image, feedback_data)
                            self.draw_stats_panel(image)
                        
                        else:
                            # No pose detected
                            h, w, _ = image.shape
                            cv2.putText(image, "Step into the camera view", (w//2 - 200, h//2),
                                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (100, 100, 255), 2)
                            
                            self.draw_top_bar(image, 0)
                            self.draw_stats_panel(image)
                    
                    except Exception as e:
                        print(f"Error: {e}")
                
                # Update session time
                self.total_session_time = time.time() - session_start
                
                # Show window
                cv2.imshow('Yoga Pose Detector - Professional Edition', image)
                
                # Handle keys
                key = cv2.waitKey(10) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord(' '):
                    self.show_pose_menu = not self.show_pose_menu
                elif key == ord('r'):
                    self.pose_hold_time = 0
                    self.is_holding_pose = False
                    self.pose_count = 0
                    self.best_accuracy = 0
                    session_start = time.time()
                    print("✓ Stats reset")
                elif key in [ord('1'), ord('2'), ord('3'), ord('4'), ord('5')]:
                    pose_idx = int(chr(key)) - 1
                    if pose_idx < len(self.reference_poses):
                        self.current_pose = list(self.reference_poses.keys())[pose_idx]
                        self.is_holding_pose = False
                        self.pose_hold_time = 0
                        self.show_pose_menu = False
                        print(f"✓ Switched to: {self.current_pose}")
        
        cap.release()
        cv2.destroyAllWindows()
        
        print(f"\n✅ Session Summary")
        print(f"=" * 50)
        print(f"Total Duration: {int(self.total_session_time)}s")
        print(f"Poses Held: {self.pose_count}")
        print(f"Best Accuracy: {int(self.best_accuracy * 100)}%")
        print(f"=" * 50)

if __name__ == "__main__":
    app = YogaPoseDetector()
    app.run()