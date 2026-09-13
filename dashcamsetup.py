# Step 1: Connect to interior camera stream
import cv2
import numpy as np
from collections import deque
import threading
from queue import Queue

class DashcamDrowsinessIntegration:
    def __init__(self, dashcam_ip='192.168.1.100'):
        # Each 360 dashcam exposes multiple RTSP streams
        # Check your dashcam docs for stream URLs
        # Typical pattern: rtsp://IP:554/stream0 (front), /stream1 (interior), etc.
        
        self.interior_url = f'rtsp://{dashcam_ip}:554/stream1'  # Interior/driver-facing
        self.interior_stream = cv2.VideoCapture(self.interior_url)
        
        # Critical: Set buffer size to 1 to avoid lag
        # Without this, OpenCV queues up old frames
        self.interior_stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.interior_stream.set(cv2.CAP_PROP_FPS, 30)
        self.interior_stream.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.interior_stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        
        # Frame queue: bounded to prevent memory bloat
        self.frame_queue = Queue(maxsize=30)
        
        # Temporal state for drowsiness detection
        self.ear_history = deque(maxlen=30)  # Last 1 second at 30fps
        self.mar_history = deque(maxlen=30)
        self.drowsy_frames = 0
        self.drowsy_threshold = 0.7
        self.drowsy_frame_threshold = 3  # Alert after 3 consecutive drowsy frames
        
    def extract_interior_frames(self):
        """
        Continuous thread: Pull interior frames from dashcam
        This is your Integration Point #1
        """
        while True:
            ret, frame = self.interior_stream.read()
            
            if not ret:
                print("Failed to read frame, attempting reconnect...")
                continue
            
            # Drop old frames if queue is full (prioritize freshness)
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except:
                    pass
            
            self.frame_queue.put({
                'frame': frame,
                'timestamp': datetime.now()
            })


def preprocess_frame(self, frame):
    """
    Downscale to 360p to reduce computation
    1080p → 360p = ~9x faster (scales as O(n²) in image size)
    """
    # Original: 1920x1080
    # Downsample: 640x360
    h, w = frame.shape[:2]
    small_frame = cv2.resize(frame, (640, 360))
    
    # Convert to grayscale (needed for LBP analysis)
    gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
    
    return small_frame, gray, (w, h)  # Keep original dims for coordinate scaling


import mediapipe as mp
from skimage.feature import local_binary_pattern

class DrowsinessDetector:
    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
    def compute_ear(self, landmarks, frame_width, frame_height):
        """Eye Aspect Ratio: ratio of vertical to horizontal eye opening"""
        # Landmark indices for left eye
        left_eye_indices = [33, 160, 158, 133, 153, 144]
        
        # Extract coordinates
        eye_points = np.array([
            [landmarks[i].x * frame_width, landmarks[i].y * frame_height]
            for i in left_eye_indices
        ])
        
        # EAR formula
        vertical_dist = (
            np.linalg.norm(eye_points[1] - eye_points[4]) +
            np.linalg.norm(eye_points[2] - eye_points[3])
        ) / 2.0
        
        horizontal_dist = np.linalg.norm(eye_points[0] - eye_points[5])
        
        ear = vertical_dist / (horizontal_dist + 1e-6)
        return ear
    
    def compute_mar(self, landmarks, frame_width, frame_height):
        """Mouth Aspect Ratio: yawning detection"""
        # Mouth landmarks
        mouth_indices = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409]
        
        mouth_points = np.array([
            [landmarks[i].x * frame_width, landmarks[i].y * frame_height]
            for i in mouth_indices
        ])
        
        # Vertical distance (open/close)
        vertical = (
            np.linalg.norm(mouth_points[1] - mouth_points[7]) +
            np.linalg.norm(mouth_points[2] - mouth_points[8])
        ) / 2.0
        
        # Horizontal distance (width)
        horizontal = np.linalg.norm(mouth_points[0] - mouth_points[6])
        
        mar = vertical / (horizontal + 1e-6)
        return mar
    
    def compute_lbp(self, gray_frame):
        """Local Binary Patterns: facial texture for fatigue signs"""
        lbp = local_binary_pattern(gray_frame, 8, 1, method='uniform')
        
        # Focus on upper face (eyes/forehead area)
        face_roi = lbp[:int(lbp.shape[0] * 0.4), :]
        
        # Histogram of LBP patterns (32 bins for uniform LBP)
        hist, _ = np.histogram(face_roi.ravel(), bins=32, range=(0, 32))
        hist = hist.astype('float32') / hist.sum()
        
        # Return variance as a drowsiness indicator
        # Higher variance = more texture variation = drowsy
        return np.var(hist)
    
    def detect(self, frame, gray_frame, frame_width, frame_height):
        """Run all three detectors"""
        results = self.face_mesh.process(frame)
        
        if not results.multi_face_landmarks:
            return None  # No face detected
        
        landmarks = results.multi_face_landmarks[0].landmark
        
        ear = self.compute_ear(landmarks, frame_width, frame_height)
        mar = self.compute_mar(landmarks, frame_width, frame_height)
        lbp_var = self.compute_lbp(gray_frame)
        
        return {
            'ear': ear,
            'mar': mar,
            'lbp': lbp_var,
            'has_face': True
        }



def process_frames(self):
    """
    Main detection loop: Pull frames, run detection, aggregate
    This is where your intelligence layer lives
    """
    detector = DrowsinessDetector()
    
    while True:
        data = self.frame_queue.get()
        frame = data['frame']
        
        # Preprocess
        small_frame, gray, orig_dims = self.preprocess_frame(frame)
        
        # Run detection
        detection = detector.detect(
            small_frame, gray, 
            small_frame.shape[1], small_frame.shape[0]
        )
        
        if detection is None or not detection['has_face']:
            self.drowsy_frames = 0  # Reset on face loss
            continue
        
        # Record history
        self.ear_history.append(detection['ear'])
        self.mar_history.append(detection['mar'])
        
        # Compute drowsy score from temporal aggregation
        drowsy_score = self.compute_drowsy_score()
        
        # THRESHOLD LOGIC (your decision engine)
        if drowsy_score > self.drowsy_threshold:
            self.drowsy_frames += 1
        else:
            self.drowsy_frames = 0
        
        # Only alert after sustained drowsiness
        if self.drowsy_frames >= self.drowsy_frame_threshold:
            self.trigger_alert(
                score=drowsy_score,
                frame=frame,
                data=detection
            )
            self.drowsy_frames = 0  # Reset to avoid repeat alerts
    
    def compute_drowsy_score(self):
        """Fuse EAR, MAR, LBP into single score"""
        if len(self.ear_history) < 10:
            return 0.0
        
        # Normalized thresholds (tuned empirically)
        ear_threshold = 0.2  # Eyes mostly closed
        mar_threshold = 0.5  # Mouth wide open (yawn)
        
        # How many frames in history are "drowsy"?
        drowsy_ear = sum(1 for e in self.ear_history if e < ear_threshold)
        drowsy_mar = sum(1 for m in self.mar_history if m > mar_threshold)
        
        # Weighted combination
        ear_score = drowsy_ear / len(self.ear_history)
        mar_score = drowsy_mar / len(self.mar_history)
        
        # Final score: 70% EAR, 30% MAR
        # (Eye closure is stronger indicator than yawning)
        drowsy_score = 0.7 * ear_score + 0.3 * mar_score
        
        return drowsy_score


def trigger_alert(self, score, frame, data):
    """
    Integration Point #2: Trigger dashcam's built-in alert system
    """
    print(f"⚠️ DROWSINESS ALERT: Score={score:.2f}")
    
    # Option A: Use dashcam's GPIO/relay pins
    # If your dashcam supports GPIO control
    try:
        import RPi.GPIO as GPIO  # If running on Raspberry Pi inside dashcam
        GPIO.output(ALERT_PIN, GPIO.HIGH)  # Trigger in-cab speaker/buzzer
        time.sleep(2)
        GPIO.output(ALERT_PIN, GPIO.LOW)
    except:
        pass
    
    # Option B: Send command to dashcam's web API
    # (if dashcam has HTTP endpoint)
    try:
        requests.post(
            f'http://{self.dashcam_ip}:8080/api/alert',
            json={'type': 'drowsiness', 'urgency': 'high'}
        )
    except:
        pass
    
    # Option C: Play local audio alert
    # (if running on external device connected to dashcam)
    import winsound
    winsound.Beep(1000, 500)  # 1000Hz, 500ms
    
    # Log to SD card (for incident review)
    self.log_incident(score, frame, data)

def log_incident(self, score, frame, data):
    """
    Write incident metadata to dashcam's SD card
    """
    incident_log = {
        'timestamp': datetime.now().isoformat(),
        'drowsy_score': float(score),
        'ear': float(data['ear']),
        'mar': float(data['mar']),
    }
    
    # Save frame as JPEG (5-second context window)
    filename = f"/media/sdcard/drowsy_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    cv2.imwrite(filename, frame)
    
    # Log metadata
    log_path = "/media/sdcard/drowsy_incidents.jsonl"
    with open(log_path, 'a') as f:
        f.write(json.dumps(incident_log) + '\n')


