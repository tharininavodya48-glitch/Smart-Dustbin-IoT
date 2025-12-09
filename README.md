# Smart Dustbin with AI Waste Classification

## Project Overview
This project simulates a Smart Dustbin that automatically sorts waste into 4 categories (Blue, Red, Green, Black) using Artificial Intelligence and IoT concepts. It uses computer vision to detect waste and sends signals to an Arduino to open the correct bin.

📸 Circuit Diagram
![Final Circuit Diagram](Circuit_Diagram_Final_With_Sensors.jpg)

## 🧠 AI Model (YOLOv8)
The trained AI model file (`best.pt`) is too large to host directly on GitHub.
👉 [**Click Here to Download the best.pt Model**](https://drive.google.com/file/d/1VV4awMex0zKLttXkjVrxf3-_jf_PhQcw/view?usp=drive_link)

## 🛠️ How It Works
1. **Detection:** An Ultrasonic Sensor (simulated via Potentiometer) detects when a user approaches.
2. **Classification:** A camera captures the waste, and a Python script using **YOLOv8** classifies it.
3. **Sorting:** The system sends a signal to an **Arduino**, which rotates the specific **Servo Motor** to open the correct bin lid.

## 💻 Tech Stack
- **Python:** YOLOv8, OpenCV, PyQt6, PySerial.
- **Hardware Simulation:** Proteus 8 Professional.
- **Microcontroller:** Arduino Uno (Simulino).

## 🚀 How to Run
1. Install Python libraries:
   ```bash
   pip install ultralytics opencv-python pyserial PyQt6

2. Download the best.pt model from the link above and place it in the project folder.
3. Open withsensor.pdsprj in Proteus.
4. Run the Python script:
    python final.py
