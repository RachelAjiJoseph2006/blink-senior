import sys
import cv2
import mediapipe as mp
import socket
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QLabel, QPushButton, QWidget, QStackedWidget, QHBoxLayout
)
from PyQt5.QtCore import QThread, pyqtSignal, QTimer
from collections import deque
import time
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas


ESP_IP = "192.168.137.140"  # Change to your ESP32 IP
ESP_PORT = 12345  # UDP port


class UDPClient:
    def __init__(self, ip, port):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.esp_address = (ip, port)


    def send_command(self, command):
        self.sock.sendto(command.encode(), self.esp_address)
        print(f"Sent command: {command}")


udp_client = UDPClient(ESP_IP, ESP_PORT)


class EyeBlinkDetector(QThread):
    blink_detected = pyqtSignal(int)  # Emit the number of blinks detected (1 or 2)
    frame_ready = pyqtSignal(object)  # Emit the frame to be displayed


    def __init__(self, camera_url):
        super().__init__()
        self.camera_url = camera_url
        self.running = True
        self.blink_threshold = 0.5  # EAR threshold for blink detection
        self.double_blink_max_time = 400  # Max time (ms) for double blink
        self.last_blink_time = None
        self.single_blink_emitted = False
        self.ear_history = deque(maxlen=5)


        # Initialize MediaPipe FaceMesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )


    def run(self):
        cap = cv2.VideoCapture(self.camera_url)
        if not cap.isOpened():
            print("Error: Could not access the camera.")
            return


        print("Camera accessed successfully.")


        while self.running:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read a frame.")
                break


            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)


            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    left_eye_ratio = self.calculate_eye_ratio(
                        face_landmarks.landmark, [33, 160, 159, 158, 153, 144, 145, 133]
                    )
                    right_eye_ratio = self.calculate_eye_ratio(
                        face_landmarks.landmark, [362, 385, 387, 386, 374, 380, 381, 263]
                    )


                    smoothed_ear = (left_eye_ratio + right_eye_ratio) / 2
                    self.ear_history.append(smoothed_ear)
                    self.detect_blink(sum(self.ear_history) / len(self.ear_history))

            cv2.imshow("LIVE VIDEO",frame)
            self.frame_ready.emit(rgb_frame)


        cap.release()


    def detect_blink(self, smoothed_ear):
        current_time = time.time() * 1000  # Current time in milliseconds


        if smoothed_ear < self.blink_threshold:  # Eye closed
            if self.last_blink_time is None:
                self.last_blink_time = current_time
                self.single_blink_emitted = False


        elif smoothed_ear >= self.blink_threshold:  # Eye open
            if self.last_blink_time is not None:
                elapsed_time = current_time - self.last_blink_time
                if elapsed_time < self.double_blink_max_time:
                    if not self.single_blink_emitted:
                        self.blink_detected.emit(1)  # Single blink detected
                        print("Single Blink Detected!")
                        self.send_motor_command("0")
                        self.single_blink_emitted = True
                else:
                    if self.single_blink_emitted:
                        self.blink_detected.emit(2)  # Double blink detected
                        print("Double Blink Detected!")
                        self.send_motor_command("1")
                self.last_blink_time = None


    def stop(self):
        self.running = False
        self.wait()


    @staticmethod
    def calculate_eye_ratio(landmarks, indices):
        horizontal_distance = (
            (landmarks[indices[0]].x - landmarks[indices[4]].x) ** 2
            + (landmarks[indices[0]].y - landmarks[indices[4]].y) ** 2
        ) ** 0.5
        vertical_distance_1 = (
            (landmarks[indices[1]].x - landmarks[indices[5]].x) ** 2
            + (landmarks[indices[1]].y - landmarks[indices[5]].y) ** 2
        ) ** 0.5
        vertical_distance_2 = (
            (landmarks[indices[2]].x - landmarks[indices[6]].x) ** 2
            + (landmarks[indices[2]].y - landmarks[indices[6]].y) ** 2
        ) ** 0.5


        eye_ratio = (vertical_distance_1 + vertical_distance_2) / (2.0 * horizontal_distance)
        return eye_ratio
    
    def send_motor_command(self, command):
        udp_client.send_command(command)





class EyeControlGUI(QMainWindow):
    def __init__(self, camera_url):
        super().__init__()


        self.setWindowTitle("Eye Control GUI")
        self.setGeometry(100, 100, 800, 600)


        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)


        self.layout = QHBoxLayout()
        self.main_widget.setLayout(self.layout)


        self.left_layout = QVBoxLayout()
        self.right_layout = QVBoxLayout()
        self.layout.addLayout(self.left_layout)
        self.layout.addLayout(self.right_layout)


        self.label = QLabel("Welcome! Blink to navigate.")
        self.left_layout.addWidget(self.label)


        self.blink_counter_label = QLabel("Blinks: 0")
        self.left_layout.addWidget(self.blink_counter_label)


        self.wake_counter_label = QLabel("Wake Blinks: 0")
        self.left_layout.addWidget(self.wake_counter_label)


        self.awake_status_label = QLabel("Awake: OFF")
        self.left_layout.addWidget(self.awake_status_label)


        self.options_stack = QStackedWidget()
        self.left_layout.addWidget(self.options_stack)


        # Main Options
        self.main_options = QWidget()
        self.main_layout = QVBoxLayout()
        self.main_options.setLayout(self.main_layout)


        self.sideways_button = QPushButton("Sideways")
        self.main_layout.addWidget(self.sideways_button)


        self.upper_body_button = QPushButton("Upper Body")
        self.main_layout.addWidget(self.upper_body_button)


        self.lower_body_button = QPushButton("Lower Body")
        self.main_layout.addWidget(self.lower_body_button)


        self.options_stack.addWidget(self.main_options)


        # Sub-options
        self.sub_options_widgets = {
            "Sideways": self.create_sub_options(["Right", "Left", "Back"], ["motor3/right", "motor3/left", "stop"]),
            "Upper Body": self.create_sub_options(["0°","30°", "45°", "60°", "Back"], ["0","30","45","60","-1"]),
            "Lower Body": self.create_sub_options(["0°","30°", "45°", "60°", "Back"], ["0","30","45","60","-1"])
        }


        self.option_list = [self.sideways_button, self.upper_body_button, self.lower_body_button]
        self.current_index = 0
        self.blink_counter = 0
        self.wake_counter = 0
        self.awake = False
        self.highlight_option(self.main_options)  # Start with the first option highlighted


        # Initialize sub-option highlighted states
        self.sub_option_highlighted = {
            "Sideways": 0,
            "Upper Body": 0,
            "Lower Body": 0
        }


        self.eye_blink_detector = EyeBlinkDetector(camera_url)
        self.eye_blink_detector.blink_detected.connect(self.handle_blink)
        self.eye_blink_detector.frame_ready.connect(self.update_frame)
        self.eye_blink_detector.start()


        # Initialize matplotlib figure and axis
        self.fig, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.fig)
        self.right_layout.addWidget(self.canvas)
        self.im = None


        # Use QTimer to periodically update the matplotlib plot
        self.timer = QTimer()
        self.timer.timeout.connect(self.refresh_plot)
        self.timer.start(50)


        # Timer to reset blink counter
        self.reset_timer = QTimer()
        self.reset_timer.timeout.connect(self.reset_blink_counter)
        self.reset_timer.start(5000)


        # Timer to turn off awake flag
        self.awake_timer = QTimer()
        self.awake_timer.timeout.connect(self.turn_off_awake)


    def create_sub_options(self, options, commands):
        widget = QWidget()
        layout = QVBoxLayout()
        widget.setLayout(layout)
        self.options_stack.addWidget(widget)


        for option, command in zip(options, commands):
            button = QPushButton(option)
            button.clicked.connect(lambda checked, cmd=command: self.send_motor_command(cmd))
            layout.addWidget(button)


        return widget


    def handle_blink(self, blink_count):
        if not self.awake:
            self.wake_counter += blink_count
            self.wake_counter_label.setText(f"Wake Blinks: {self.wake_counter}")
            if self.wake_counter >= 5:
                self.awake = True
                self.awake_status_label.setText("Awake: ON")
                self.wake_counter = 0
                self.awake_timer.start(10000)  # Start timer to turn off awake flag after 10 seconds
        else:
            self.blink_counter += blink_count
            self.blink_counter_label.setText(f"Blinks: {self.blink_counter}")
            if self.blink_counter == 1:
                current_widget = self.options_stack.currentWidget()
                if isinstance(current_widget, QWidget):
                    self.highlight_option(current_widget)
            elif self.blink_counter > 1:
                self.select_option()
                self.blink_counter = 0
            self.awake_timer.start(10000)  # Reset timer to turn off awake flag after 10 seconds


    def highlight_option(self, current_widget):
        buttons = current_widget.findChildren(QPushButton)
        for i, button in enumerate(buttons):
            if self.current_index == i:
                button.setStyleSheet("background-color: yellow;")
            else:
                button.setStyleSheet("")


        self.current_index = (self.current_index + 1) % len(buttons)


    def select_option(self):
        current_widget = self.options_stack.currentWidget()
        if isinstance(current_widget, QWidget):
            buttons = current_widget.findChildren(QPushButton)
            selected_button = buttons[self.current_index - 1]  # Adjust for cycling
            selected_text = selected_button.text()
            print(f"Selected: {selected_text}")


            # Map selected text to ESP32 commands
            command_mapping = {
                "Sideways": "1",
                "Upper Body": "2",
                "Lower Body": "3",
                "Right": "motor3/right",
                "Left": "motor3/left",
                "0°":"0",
                "30°":"30",
                "45°":"45",
                "60°":"60",
                "Back": "stop"
            }


            if selected_text in command_mapping:
                command = command_mapping[selected_text]
                print(f"Sending command: {command}")  # Debugging statement
                self.send_motor_command(command)


            # Handle navigation
            if selected_text == "Back":
                self.options_stack.setCurrentWidget(self.main_options)
                self.current_index = 0
                self.highlight_option(self.main_options)
            elif selected_text in self.sub_options_widgets:
                self.options_stack.setCurrentWidget(self.sub_options_widgets[selected_text])
                self.current_index = 0
                self.highlight_option(self.sub_options_widgets[selected_text])


    def send_motor_command(self, command):
        udp_client.send_command(command)


    def update_frame(self, rgb_frame):
        if self.im is None:
            self.im = self.ax.imshow(rgb_frame)
        else:
            self.im.set_array(rgb_frame)


    def refresh_plot(self):
        self.canvas.draw()


    def reset_blink_counter(self):
        self.blink_counter = 0
        self.blink_counter_label.setText("Blinks: 0")
        self.wake_counter = 0
        self.wake_counter_label.setText("Wake Blinks: 0")


    def turn_off_awake(self):
        self.awake = False
        self.awake_status_label.setText("Awake: OFF")


    def closeEvent(self, event):
        self.eye_blink_detector.stop()
        event.accept()




if __name__ == "__main__":
    app = QApplication(sys.argv)
    camera_url = 0  # Default camera
    gui = EyeControlGUI(camera_url)
    gui.show()
    sys.exit(app.exec_())