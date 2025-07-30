# Face Recognition System with Arduino Integration

This project implements a face recognition system using a webcam or IP camera, integrated with an Arduino to display recognition results on an LCD and control an LED. The system logs attendance events and manages user face encodings for real-time verification.

## Features

- **Face Registration**: Capture and save face encodings for new users.
- **Face Verification**: Detect and recognize faces, logging entry/exit events.
- **Arduino Integration**: Display recognition results on a 16x2 I2C LCD and control an LED.
- **Attendance Logging**: Save events to a CSV file with timestamps, usernames, status, and duration.
- **User Management**: List, delete, and manage registered users.
- **Blurry Image Detection**: Ensure high-quality face captures.
- **Daily Attendance Tracking**: Track total presence time per user per day.

## Prerequisites

- **Hardware**:
  - Windows PC
  - Webcam or IP camera
  - Arduino board (e.g., Arduino Uno)
  - 16x2 I2C LCD display (address 0x16)
  - LED with 220Ω resistor
  - USB cable for Arduino
- **Software**:
  - Python 3.7–3.11
  - Arduino IDE
  - Git

## Setup Instructions

### Python Environment Setup

1. **Install Python**:

   - Download and install Python from python.org.
   - Ensure "Add Python to PATH" is checked during installation.
   - Verify installation:

     ```bash
     python --version
     pip --version
     ```

2. **Clone the Repository**:

   ```bash
   git clone https://github.com/ahsan-bajwaa/face_recognition_system.git
   cd face-recognition-system
   ```

3. **Create a Virtual Environment**:

   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

4. **Install Dependencies**:

   - Ensure `requirements.txt` is in the project directory.
   - Install libraries:

     ```bash
     pip install -r requirements.txt
     ```
   - If `dlib` installation fails, install Microsoft Visual C++ Build Tools from visualstudio.microsoft.com and try:

     ```bash
     pip install cmake
     pip install dlib
     pip install face_recognition
     ```

5. **Update Configuration**:

   - In `face_recog_mobile_3.5.py`, update `IP_ADDRESS` to your camera's address or `0` for the default webcam.
   - Update `ARDUINO_PORT` to your Arduino's COM port (e.g., `COM3`). Find the port in Device Manager under "Ports (COM & LPT)".

6. **Run the Program**:

   ```bash
   python face_recog_mobile.py
   ```

   - Follow the menu to register users, verify faces, or manage logs.

### Arduino Setup

1. **Install Arduino IDE**:

   - Download from arduino.cc.
   - Install and open the IDE.

2. **Connect Hardware**:

   - **LCD Wiring**:
     - VCC to Arduino 5V
     - GND to Arduino GND
     - SDA to Arduino A4
     - SCL to Arduino A5
   - **LED Wiring**:
     - LED anode to Arduino pin 8 via 220Ω resistor
     - LED cathode to Arduino GND
   - Connect the Arduino to your PC via USB.

3. **Install Arduino Libraries**:

   - Open Arduino IDE, go to `Sketch > Include Library > Manage Libraries`.
   - Install `LiquidCrystal I2C` by Frank de Brabander.

4. **Upload Arduino Code**:

   - Open `face_recog_mobile_Arduino.txt` in Arduino IDE (save as `.ino`).
   - Select your board (`Tools > Board > Arduino Uno`) and port (`Tools > Port`).
   - Upload the code.

5. **Test the System**:

   - Run the Python script and select option `2` to verify faces.
   - The LCD should display "Waiting for data", "Unknown/No Match", or "Face Matched!" with the username and status. The LED lights up on a match.

## Project Structure

```
face-recognition-system/
│
├── face_recog_mobile.py         # Main Python script for face recognition
├── face_recog_mobile_Arduino.ino # Arduino code for LCD and LED
├── requirements.txt                 # Python dependencies
├── .gitignore                       # Git ignore file
├── face_logs/                       # Directory for attendance logs
├── face_encodings/                  # Directory for face encodings
```

## Usage

- **Register a User**: Select option `1`, enter a username, and follow prompts to capture face encodings.
- **Verify Faces**: Select option `2` to start face recognition. Press `q` to stop.
- **Manage Users**: Use options `3` and `4` to list or delete users.
- **View/Clear Logs**: Use options `5` and `6` to view or clear attendance logs.

## Troubleshooting

- **Camera Issues**:
  - Verify the `IP_ADDRESS` in the Python script.
  - Test with a local webcam (`IP_ADDRESS = 0`).
- **Arduino Issues**:
  - Ensure the correct COM port is set.
  - Close Arduino IDE Serial Monitor before running the Python script.
  - Check LCD wiring and I2C address (0x27).
- **Python Library Issues**:
  - Ensure `dlib` and `face_recognition` are installed correctly.
  - Use Python 3.7–3.11 for compatibility.

## License

This project is licensed under Aaa.