import os
import cv2
import pickle
import face_recognition
import numpy as np
from datetime import datetime, timedelta
from collections import deque
from scipy.spatial.distance import euclidean
import csv
import time
import serial
import re
import serial.tools.list_ports

# Constants
IP_ADDRESS = "http://Aaa:654321@192.168.29.158:7788//video"
LOG_DIR = "face_logs"
ENCODINGS_DIR = "face_encodings"
TARGET_WIDTH = 1280
TARGET_HEIGHT = 720
FRAME_SKIP = 5  # Reduced to process more frames
FACE_DISTANCE_THRESHOLD = 0.45
MIN_CONFIDENCE = 0.6
MIN_ENCODINGS_TO_SAVE = 5
MAX_ENCODINGS_PER_USER = 8
BOX_DISPLAY_DURATION = 1.0
ENCODING_INTERVAL = 0.5
STATE_CHANGE_THRESHOLD = timedelta(seconds=5)  # 5s for entry/exit
PRESENCE_THRESHOLD = timedelta(seconds=5)  # 5s absence for exit
ARDUINO_BAUD = 9600
ARDUINO_PORT = "/dev/ttyACM0"
UPDATE_INTERVAL = 1.0  # 1s update interval
FACE_TIMEOUT = 2.0  # 2s timeout before reverting to waiting

# Arduino Serial Setup
ser = None
last_send_time = 0
last_match_username = None
last_match_status = None
last_no_face_time = time.time()  # Initialize globally to avoid scoping issue

# Global state for tracking user presence and daily attendance
user_presence = {}
daily_attendance = {}  # Tracks total presence time per user per day
current_date = datetime.now().date()

def initialize_serial():
    global ser
    if ser and ser.is_open:
        ser.close()
    try:
        print(f"[INFO] Attempting to connect to Arduino on {ARDUINO_PORT}")
        ser = serial.Serial(ARDUINO_PORT, ARDUINO_BAUD, timeout=1)
        ser.flushInput()
        ser.flushOutput()
        print(f"[INFO] Connected to Arduino on {ARDUINO_PORT}")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to connect to Arduino on {ARDUINO_PORT}: {str(e)}")
        print("Available ports:")
        for port in serial.tools.list_ports.comports():
            print(f" - {port.device}: {port.description}")
        print("Please check:")
        print("1. Is the Arduino connected?")
        print("2. Is the Arduino IDE's Serial Monitor closed?")
        print("3. Do you have permission? (Run: sudo usermod -a -G dialout $USER)")
        ser = None
        return False

initialize_serial()

def initialize_directories():
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(ENCODINGS_DIR, exist_ok=True)

def initialize_log_file():
    log_file = os.path.join(LOG_DIR, "attendance_logs.csv")
    if not os.path.exists(log_file):
        with open(log_file, mode="w", newline="") as f:
            csv.writer(f).writerow(["Timestamp", "Username", "Status", "Confidence", "Duration", "Total Presence (HH:MM:SS)"])

def seconds_to_hms(seconds):
    """Convert seconds to HH:MM:SS format."""
    hours, remainder = divmod(int(seconds), 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def reset_daily_attendance_if_new_day():
    global current_date, daily_attendance
    today = datetime.now().date()
    if today != current_date:
        print(f"[INFO] New day detected ({today}), resetting daily attendance.")
        daily_attendance = {}
        current_date = today

def send_to_arduino(username, status):
    global ser, last_send咯, last_match_username, last_match_status
    current_time = time.time()
    
    if not ser or not ser.is_open:
        print("[WARNING] Serial port not open, attempting to reconnect...")
        if not initialize_serial():
            print("[ERROR] Cannot send to Arduino: no connection")
            return
    
    try:
        message = f"{username}:{status}\n"
        ser.flushInput()
        ser.write(message.encode('utf-8'))
        ser.flushOutput()
        print(f"[INFO] Sent to Arduino: {message.strip()}")
        
        last_send_time = current_time
        if status in ["entered", "exited"]:
            last_match_username = username
            last_match_status = status
            
    except Exception as e:
        print(f"[ERROR] Failed to send to Arduino: {str(e)}")
        ser.close()
        ser = None
        initialize_serial()

def log_face_event(username, status, confidence, duration=None):
    reset_daily_attendance_if_new_day()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_file = os.path.join(LOG_DIR, "attendance_logs.csv")
    total_presence = daily_attendance.get(username, 0)
    
    if status == "exited" and duration:
        daily_attendance[username] = total_presence + duration
    
    try:
        with open(log_file, mode="a", newline="") as f:
            csv.writer(f).writerow([
                timestamp,
                username,
                status,
                f"{confidence:.3f}",
                f"{duration:.1f}s" if duration else "",
                seconds_to_hms(total_presence)
            ])
        send_to_arduino(username, status)
    except Exception as e:
        print(f"[ERROR] Failed to write to log file: {str(e)}")

def load_local_encodings():
    known_data = {}
    try:
        for filename in os.listdir(ENCODINGS_DIR):
            if filename.endswith(".pkl"):
                username = os.path.splitext(filename)[0]
                with open(os.path.join(ENCODINGS_DIR, filename), "rb") as f:
                    encodings = pickle.load(f)
                    if encodings:
                        known_data[username] = encodings
                        print(f"[INFO] Loaded {len(encodings)} encodings forhealthy {username}")
        return known_data
    except Exception as e:
        print(f"[ERROR] Failed to load local encodings: {str(e)}")
        return {}

def is_blurry(image, face_location, threshold=35.0):
    top, right, bottom, left = face_location
    roi = image[max(0, top-50):bottom+50, max(0, left-50):right+50]
    if roi.size == 0:
        return True
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F).var()
    return lap < threshold

def is_encoding_unique(new_enc, enc_list, threshold=0.4):
    return all(euclidean(new_enc, existing) > threshold for existing in enc_list)

def has_facial_landmarks(rgb_small, face_location):
    scaled_location = [(t//2, r//2, b//2, l//2) for t, r, b, l in [face_location]]
    landmarks = face_recognition.face_landmarks(rgb_small, scaled_location)
    return bool(landmarks and all(key in landmarks[0] for key in ['left_eye', 'right_eye', 'nose_bridge']))

def capture_and_save_face(username):
    cap = cv2.VideoCapture(IP_ADDRESS)
    if not cap.isOpened():
        print("[ERROR] Couldn't open webcam")
        return False

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, TARGET_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, TARGET_HEIGHT)
    encodings = deque(maxlen=MAX_ENCODINGS_PER_USER)
    frame_count = 0
    last_box_time = 0
    box_coordinates = None
    last_encoding_time = 0
    check_landmarks = True

    print(f"[INFO] Capturing face encodings for {username}. Move head slowly. Press 'q' to stop.")
    cv2.namedWindow("Register Face", cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to read frame from camera.")
            break

        frame_count += 1
        current_time = time.time()
        processed_frame = frame.copy()

        if frame_count % FRAME_SKIP == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_small = cv2.resize(rgb, (0, 0), fx=0.33, fy=0.33)
            boxes = face_recognition.face_locations(rgb_small, model="hog", number_of_times_to_upsample=1)
            boxes = [(top*3, right*3, bottom*3, left*3) for (top, right, bottom, left) in boxes]

            if len(boxes) == 1:
                if is_blurry(frame, boxes[0]):
                    cv2.putText(processed_frame, "Image too blurry, move closer or improve lighting",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                elif check_landmarks and not has_facial_landmarks(rgb_small, boxes[0]):
                    cv2.putText(processed_frame, "No valid face detected",
                                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                elif current_time - last_encoding_time > ENCODING_INTERVAL:
                    new_encs = face_recognition.face_encodings(
                        rgb_small,
                        [(top//3, right//3, bottom//3, left//3) for (top, right, bottom, left) in boxes]
                    )
                    if new_encs:
                        enc = new_encs[0]
                        if len(encodings) < MIN_ENCODINGS_TO_SAVE or is_encoding_unique(enc, encodings):
                            encodings.append(enc)
                            last_encoding_time = current_time
                            print(f"[INFO] Captured encoding {len(encodings)}/{MAX_ENCODINGS_PER_USER}")
                        else:
                            cv2.putText(processed_frame, "Move head for more diverse encodings",
                                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                        box_coordinates = (boxes[0], username)
                        last_box_time = current_time

                check_landmarks = not check_landmarks

        if box_coordinates and (current_time - last_box_time) < BOX_DISPLAY_DURATION:
            (top, right, bottom, left), name = box_coordinates
            cv2.rectangle(processed_frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(processed_frame, name, (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(processed_frame, f"Encodings: {len(encodings)}/{MAX_ENCODINGS_PER_USER}",
                        (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(processed_frame, "Move head slowly in all directions",
                        (10, TARGET_HEIGHT - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        processed_frame = cv2.resize(processed_frame, (0, 0), fx=0.7, fy=0.7)
        cv2.imshow("Register Face", processed_frame)

        if (cv2.getWindowProperty("Register Face", cv2.WND_PROP_VISIBLE) < 1 or
                cv2.waitKey(1) & 0xFF == ord('q') or
                len(encodings) >= MAX_ENCODINGS_PER_USER):
            break

    cap.release()
    cv2.destroyAllWindows()

    if len(encodings) >= MIN_ENCODINGS_TO_SAVE:
        print(f"[INFO] Captured {len(encodings)} encodings for {username}")
        return save_local_encodings(username, encodings)
    else:
        print(f"[ERROR] Need at least {MIN_ENCODINGS_TO_SAVE} distinct encodings")
        return False

def save_local_encodings(username, encodings):
    try:
        filepath = os.path.join(ENCODINGS_DIR, f"{username}.pkl")
        with open(filepath, 'wb') as f:
            pickle.dump(list(encodings), f)
        print(f"[SUCCESS] Saved encodings to {filepath}")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to save encodings: {str(e)}")
        return False

def verify_user_face():
    global last_no_face_time
    initialize_directories()
    initialize_log_file()
    cap = cv2.VideoCapture(IP_ADDRESS)
    if not cap.isOpened():
        print("[ERROR] Couldn't open webcam.")
        return

    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, TARGET_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, TARGET_HEIGHT)

    known_faces = load_local_encodings()
    if not known_faces:
        print("[WARNING] No registered faces found. Please register users first.")
        return

    known_face_names = []
    known_face_encodings = []
    for name, encodings in known_faces.items():
        for encoding in encodings:
            if isinstance(encoding, np.ndarray):
                known_face_encodings.append(encoding)
                known_face_names.append(name)

    if not known_face_encodings:
        print("[ERROR] No valid face encodings found in the database.")
        return

    frame_count = 0
    current_detections = set()
    last_boxes = {}
    last_update_time = time.time()
    print("[INFO] Verifying faces. Press 'q' to stop.")
    cv2.namedWindow("Verification", cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to grab frame.")
            break

        frame = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT))
        frame_count += 1
        display_frame = frame.copy()
        current_time = time.time()

        # Always update display to prevent freezing
        if frame_count % FRAME_SKIP == 0:
            current_detections.clear()
            rgb_small = cv2.cvtColor(cv2.resize(frame, (0, 0), fx=0.5, fy=0.5), cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_small, model="hog", number_of_times_to_upsample=1)
            face_encodings = face_recognition.face_encodings(rgb_small, face_locations)
            face_locations = [(top*2, right*2, bottom*2, left*2) for (top, right, bottom, left) in face_locations]

            known_face_detected = False
            unknown_face_detected = False

            if not face_locations:
                if current_time - last_update_time >= UPDATE_INTERVAL:
                    send_to_arduino("", "Waiting")
                    last_update_time = current_time
                    last_no_face_time = current_time
            else:
                for (face_location, face_encoding) in zip(face_locations, face_encodings):
                    if face_encoding.size == 0 or not has_facial_landmarks(rgb_small, face_location):
                        cv2.putText(display_frame, "No valid face detected",
                                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                        continue

                    if is_blurry(frame, face_location):
                        cv2.putText(display_frame, "Image too blurry, move closer or improve lighting",
                                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                        continue

                    matches = face_recognition.compare_faces(
                        known_face_encodings,
                        face_encoding,
                        tolerance=FACE_DISTANCE_THRESHOLD
                    )
                    name = "Unknown"
                    distance = 1.0

                    if True in matches:
                        matched_indices = [i for i, match in enumerate(matches) if match]
                        face_distances = face_recognition.face_distance(
                            [known_face_encodings[i] for i in matched_indices],
                            face_encoding
                        )
                        min_distance = np.min(face_distances)
                        if min_distance < MIN_CONFIDENCE:
                            best_match_index = matched_indices[np.argmin(face_distances)]
                            name = known_face_names[best_match_index]
                            distance = min_distance
                            known_face_detected = True
                        else:
                            unknown_face_detected = True
                    else:
                        unknown_face_detected = True

                    color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                    top, right, bottom, left = face_location

                    if name != "Unknown":
                        current_detections.add(name)
                        now = datetime.now()
                        user_data = user_presence.get(name, {
                            'status': None,
                            'last_seen': None,
                            'first_seen': None,
                            'last_absent': None
                        })

                        if user_data['last_seen'] is None or (now - user_data['last_seen']).total_seconds() >= PRESENCE_THRESHOLD.total_seconds():
                            if user_data['status'] is None:
                                user_data['status'] = "entered"
                                user_data['first_seen'] = now
                                log_face_event(name, "entered", distance)
                            elif user_data['status'] == "entered":
                                user_data['status'] = "exited"
                                duration = (now - user_data['first_seen']).total_seconds()
                                log_face_event(name, "exited", 0, duration)
                                user_data['first_seen'] = None  # Reset first_seen after exit
                            elif user_data['status'] == "exited":
                                user_data['status'] = "entered"
                                user_data['first_seen'] = now
                                log_face_event(name, "entered", distance)
                            user_data['last_absent'] = None

                        user_data['last_seen'] = now
                        user_presence[name] = user_data

                    last_boxes[name] = {
                        'coords': (top, right, bottom, left),
                        'color': color,
                        'distance': distance,
                        'time': current_time
                    }

                # Update Arduino every second if face is detected
                if current_time - last_update_time >= UPDATE_INTERVAL:
                    if known_face_detected:
                        for name in current_detections:
                            status = user_presence.get(name, {}).get('status', 'unknown')
                            if status in ['entered', 'exited']:
                                send_to_arduino(name, status)
                    elif unknown_face_detected:
                        send_to_arduino("", "No Match Found")
                    last_update_time = current_time

        # Check for exited users
        for name in list(user_presence.keys()):
            if name not in current_detections:
                user_data = user_presence[name]
                last_seen = user_data.get('last_seen')
                
                if last_seen and (datetime.now() - last_seen) >= PRESENCE_THRESHOLD and user_data['status'] == "entered":
                    user_data['last_absent'] = datetime.now()
                    user_presence[name] = user_data
            else:
                user_data = user_presence.get(name)
                user_data['last_seen'] = datetime.now()
                user_data['last_absent'] = None
                user_presence[name] = user_data

        # Display face boxes
        for name, box_data in last_boxes.items():
            if current_time - box_data['time'] < BOX_DISPLAY_DURATION:
                top, right, bottom, left = box_data['coords']
                color = box_data['color']
                distance = box_data['distance']
                status = user_presence.get(name, {}).get('status', 'unknown')
                cv2.rectangle(display_frame, (left, top), (right, bottom), color, 2)
                cv2.putText(display_frame, f"{name} ({status}, {distance:.2f})",
                            (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        display_frame = cv2.resize(display_frame, (0, 0), fx=0.7, fy=0.7)
        cv2.imshow("Verification", display_frame)

        # Check for 'q' key press to exit verification
        if cv2.waitKey(1) & 0xFF == ord('q') or cv2.getWindowProperty("Verification", cv2.WND_PROP_VISIBLE) < 1:
            print("[INFO] Verification stopped by user.")
            break

    cap.release()
    cv2.destroyAllWindows()
    if ser and ser.is_open:
        ser.close()
        print("[INFO] Serial port closed.")

def list_users():
    try:
        if not os.path.exists(ENCODINGS_DIR):
            print("[INFO] No users registered yet.")
            return
        users = [os.path.splitext(f)[0] for f in os.listdir(ENCODINGS_DIR) if f.endswith(".pkl")]
        if not users:
            print("[INFO] No registered users found.")
            return
        print("\n[INFO] Registered Users:")
        for i, user in enumerate(sorted(users), 1):
            print(f"{i}. {user}")
        print(f"Total: {len(users)} users")
    except Exception as e:
        print(f"[ERROR] Failed to list users: {str(e)}")

def delete_user(username):
    try:
        filepath = os.path.join(ENCODINGS_DIR, f"{username}.pkl")
        if os.path.exists(filepath):
            os.remove(filepath)
            print(f"[INFO] Successfully deleted user '{username}'.")
            return True
        else:
            print(f"[INFO] No face data found for '{username}'.")
            return False
    except Exception as e:
        print(f"[ERROR] Failed to delete user: {str(e)}")
        return False

def view_logs():
    try:
        log_file = os.path.join(LOG_DIR, "attendance_logs.csv")
        if not os.path.exists(log_file):
            print("[INFO] No log file found.")
            return
        with open(log_file, mode="r") as f:
            reader = csv.reader(f)
            print("\n[INFO] Attendance Logs:")
            print(f"{'#':<3} | {'Timestamp':<19} | {'Username':<10} | {'Status':<8} | {'Confidence':<10} | {'Duration':<10} | {'Total Presence (HH:MM:SS)':<20}")
            print("-" * 80)
            for i, row in enumerate(reader, 1):
                if i == 1 and row[0] == "Timestamp":
                    continue
                print(f"{i:<3} | {row[0]:<19} | {row[1]:<10} | {row[2]:<8} | {row[3]:<10} | {row[4]:<10} | {row[5]:<20}")
    except Exception as e:
        print(f"[ERROR] Failed to read logs: {str(e)}")

def clear_logs():
    try:
        log_file = os.path.join(LOG_DIR, "attendance_logs.csv")
        if os.path.exists(log_file):
            os.remove(log_file)
            initialize_log_file()
            print("[INFO] Successfully cleared all logs.")
            return True
        else:
            print("[INFO] No log file found to clear.")
            return False
    except Exception as e:
        print(f"[ERROR] Failed to clear logs: {str(e)}")
        return False

def main():
    initialize_directories()
    while True:
        print("\n==== LOCAL FACE RECOGNITION SYSTEM ====")
        print("1. Register a new user")
        print("2. Verify faces (webcam)")
        print("3. List registered users")
        print("4. Delete a user")
        print("5. View attendance logs")
        print("6. Clear all logs")
        print("7. Exit")
        try:
            choice = input("Choose an option (1-7): ").strip()
            if choice == '1':
                username = input("Enter username: ").strip().lower()
                if not username:
                    print("[ERROR] Username cannot be empty")
                elif not re.match("^[a-z0-9_]+$", username):
                    print("[ERROR] Username can only contain letters, numbers and underscores")
                else:
                    capture_and_save_face(username)
            elif choice == '2':
                verify_user_face()
            elif choice == '3':
                list_users()
            elif choice == '4':
                username = input("Enter username to delete: ").strip().lower()
                if username:
                    delete_user(username)
                else:
                    print("[ERROR] Username cannot be empty")
            elif choice == '5':
                view_logs()
            elif choice == '6':
                confirm = input("Are you sure you want to clear ALL logs? (y/n): ").lower()
                if confirm == 'y':
                    clear_logs()
            elif choice == '7':
                print("Goodbye!")
                if ser and ser.is_open:
                    ser.close()
                    print("[INFO] Serial port closed.")
                break
            else:
                print("[ERROR] Invalid choice. Please enter 1-7")
        except KeyboardInterrupt:
            print("\n[INFO] Operation cancelled by user")
            if ser and ser.is_open:
                ser.close()
                print("[INFO] Serial port closed.")
        except Exception as e:
            print(f"[ERROR] An error occurred: {str(e)}")
            if ser and ser.is_open:
                ser.close()
                print("[INFO] Serial port closed.")

if __name__ == "__main__":
    try:
        face_recognition.face_encodings(np.zeros((100, 100, 3), dtype=np.uint8))
        main()
    except Exception as e:
        print(f"[CRITICAL] Face recognition library not working: {str(e)}")
        if ser and ser.is_open:
            ser.close()
            print("[INFO] Serial port closed.")
        exit(1)
