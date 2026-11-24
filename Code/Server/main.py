import os
import sys
import argparse
import struct
import time
import signal
import math
import threading
import multiprocessing
import cv2
import numpy as np
from server import Server
from message import Message_Parse
from command import Command
from led import Led
from camera import Camera
from car import Car
from buzzer import Buzzer
from Thread import stop_thread
from threading import Thread

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from PyQt5.QtWidgets import QMainWindow, QApplication
    from PyQt5.QtCore import QTimer
    from server_ui import Ui_server_ui
    _HAS_PYQT5 = True
except ImportError:
    _HAS_PYQT5 = False


class _DummyLabel:
    def __init__(self, text: str = "Server Off"):
        self._text = text

    def text(self) -> str:
        return self._text

    def setText(self, text: str) -> None:
        self._text = text


if _HAS_PYQT5:
    class _BaseWindow(QMainWindow, Ui_server_ui):
        pass
else:
    class _BaseWindow:
        pass


class mywindow(_BaseWindow):
    def __init__(self, headless: bool = False):
        self.headless = headless or (not _HAS_PYQT5)
        self.app = None
        if not self.headless:
            self.app = QApplication.instance() or QApplication(sys.argv)
            super(mywindow, self).__init__()
            self.setupUi(self)
        else:
            self.label = _DummyLabel("Server Off")
            self.Button_Server = _DummyLabel("On")

        self.ui_button_state = True
        self.config_task()

        if not self.headless:
            self.Button_Server.clicked.connect(self.on_pushButton_handle)
            self.app.lastWindowClosed.connect(self.close_application)
            self.timer = QTimer(self)
            self.timer.timeout.connect(self.check_signals)
            self.timer.start(100)

        if self.ui_button_state:
            self.on_pushButton_handle()

        signal.signal(signal.SIGINT, self.signal_handler)
        self.time_record = time.time()

    def config_task(self):
        self.tcp_server = Server()
        self.command = Command()
        self.led = Led()
        self.car = Car()
        self.buzzer = Buzzer()
        self.camera = Camera(stream_size=(400, 300))
        self.queue_cmd = multiprocessing.Queue()
        self.cmd_parse = Message_Parse()
        self.queue_led = multiprocessing.Queue()
        self.led_parse = Message_Parse()

        self.cmd_thread = None
        self.video_thread = None
        self.car_thread = None
        self.led_process = None
        self.action_process = None
        self.cmd_thread_is_running = False
        self.video_thread_is_running = False
        self.car_thread_is_running = False
        self.led_process_is_running = False
        self.action_process_is_running = False
        self.car_mode = 1
        self.rotation_flag = False
        self.send_sonic_data_time = time.time()
        self.send_light_data_time = time.time()
        self.send_line_data_time = time.time()
        self.led_mode = 0

        # Load YOLO standing/walking pedestrian models
        self.yolo_enabled = True
        self.cls_model = None
        self.det_model = None
        self.yolo_tracker = None
        self.latest_detections = []
        self._last_yolo_action = None
        try:
            from detect_pedestrian_action import PedestrianVideoTracker, load_pedestrian_models

            self.cls_model, self.det_model, cls_path, det_path = load_pedestrian_models()
            self.yolo_tracker = PedestrianVideoTracker(history_len=10)
            print(f"[YOLO] Loaded standing/walking classifier ({cls_path}) & detector ({det_path}).")
        except Exception as e:
            print(f"[YOLO] Warning: could not load YOLO models in Server main.py ({e})")

    def stop_car(self):
        self.led.colorBlink(0)
        self.camera.stop_stream()
        self.camera.close()
        self.car.close()

    def on_pushButton_handle(self):
        if self.label.text() == "Server Off":
            self.label.setText("Server On")
            self.Button_Server.setText("Off")
            self.tcp_server.start_tcp_servers()
            self.set_threading_cmd_receive(True)
            self.set_threading_video_send(True)
            self.set_threading_car_task(True)
            self.set_process_led_running(True)
        elif self.label.text() == 'Server On':
            self.label.setText("Server Off")
            self.Button_Server.setText("On")
            self.tcp_server.stop_tcp_servers()
            self.set_threading_cmd_receive(False)
            self.set_threading_video_send(False)
            self.set_threading_car_task(False)
            self.set_process_led_running(False)
            self.tcp_server = Server()

    def send_sonic_data(self):
        if time.time() - self.send_sonic_data_time > 0.5:
            self.send_sonic_data_time = time.time()
            if self.tcp_server.get_command_server_busy() == False:
                distance = self.car.sonic.get_distance()
                cmd = self.command.CMD_MODE + "#3#{:.2f}".format(distance) + "\n"
                self.tcp_server.send_data_to_command_client(cmd)
                #print(cmd)

    def send_light_data(self):
        if time.time() - self.send_light_data_time > 0.3:
            self.send_light_data_time = time.time()
            if self.tcp_server.get_command_server_busy() == False:
                adc_light_1 = self.car.adc.read_adc(0)
                adc_light_2 = self.car.adc.read_adc(1)
                cmd = self.command.CMD_MODE + "#2#{:.2f}#{:.2f}".format(adc_light_1, adc_light_2) + "\n" 
                self.tcp_server.send_data_to_command_client(cmd)
                #print(cmd)
    def send_line_data(self):
        if time.time() - self.send_line_data_time > 0.3:
            self.send_line_data_time = time.time()
            if self.tcp_server.get_command_server_busy() == False:
                ir_value_1 = self.car.infrared.read_one_infrared(1)
                ir_value_2 = self.car.infrared.read_one_infrared(2)
                ir_value_3 = self.car.infrared.read_one_infrared(3)
                cmd = self.command.CMD_MODE + "#4#{:.2f}#{:.2f}#{:.2f}".format(ir_value_1, ir_value_2, ir_value_3) + "\n"
                self.tcp_server.send_data_to_command_client(cmd)
                #print(cmd)
    def send_power_data(self):
        if self.tcp_server.get_command_server_busy() == False:
            power = self.car.adc.read_adc(2) * (3 if self.car.adc.pcb_version == 1 else 2)
            cmd = self.command.CMD_POWER + "#" + str(power) + "\n"
            self.tcp_server.send_data_to_command_client(cmd)
            #print(cmd)

    def set_threading_cmd_receive(self, state, close_time=0.3):
        if self.cmd_thread is None:
            buf_state = False
        else:
            buf_state = self.cmd_thread.is_alive()
        if state != buf_state:
            if state:
                self.cmd_thread_is_running = True
                self.cmd_thread = threading.Thread(target=self.threading_cmd_receive)
                self.cmd_thread.start()
            else:
                self.cmd_thread_is_running = False
                if self.cmd_thread is not None:
                    self.cmd_thread.join(close_time)
                    self.cmd_thread = None

    def threading_cmd_receive(self):
        while self.cmd_thread_is_running:
            cmd_queue = self.tcp_server.read_data_from_command_server()
            if cmd_queue.qsize() > 0:
                client_address, all_message = cmd_queue.get()
                main_message = all_message.strip()
                if "\n" in main_message:
                    for msg in main_message.split("\n"):
                        self.queue_cmd.put(msg)
                else:
                    self.queue_cmd.put(main_message)
            while not self.queue_cmd.empty():
                msg = self.queue_cmd.get()
                self.cmd_parse.clear_parameters()
                self.cmd_parse.parse(msg)
                print("{}".format(self.cmd_parse.input_string))
                if self.cmd_parse.command_string == self.command.CMD_LED:
                    self.queue_led.put(msg)
                elif self.cmd_parse.command_string == self.command.CMD_LED_MOD:
                    self.queue_led.put(msg)
                else:
                    if self.cmd_parse.command_string == self.command.CMD_SONIC:
                        self.send_sonic_data()
                    elif self.cmd_parse.command_string == self.command.CMD_LIGHT:
                        self.send_light_data()
                    elif self.cmd_parse.command_string == self.command.CMD_LINE:
                        self.send_line_data()
                    elif self.cmd_parse.command_string == self.command.CMD_POWER:
                        self.send_power_data()
                    elif self.cmd_parse.command_string == self.command.CMD_BUZZER:
                        self.buzzer.set_state(self.cmd_parse.int_parameter[0])
                    elif self.cmd_parse.command_string == self.command.CMD_SERVO:
                        try:
                            print(len(self.cmd_parse.int_parameter))
                            data1 = str(self.cmd_parse.int_parameter[0])
                            data2 = int(self.cmd_parse.int_parameter[1])
                            print(data1, data2)
                            if data1 == None or data2 == None:
                                continue
                            self.car.servo.set_servo_pwm(data1, data2)
                        except Exception as e:
                            print(e)
                    elif self.cmd_parse.command_string == self.command.CMD_MOTOR:
                        self.car_mode = 1
                        try:
                            duty = [int(self.cmd_parse.int_parameter[i]) for i in range(4)]
                            if duty[0] == None or duty[1]== None or duty[2] == None or duty[3] == None:
                                continue
                            scale_factor = 0.8
                            scaled_duty = [int(round(d * scale_factor)) for d in duty]
                            self.car.motor.set_motor_model(scaled_duty[0], scaled_duty[1], scaled_duty[2], scaled_duty[3])
                        except:
                            pass
                    elif self.cmd_parse.command_string == self.command.CMD_M_MOTOR:
                        self.car_mode = 1
                        duty = [int(self.cmd_parse.int_parameter[i]) for i in range(4)]
                        LX = -int((duty[1] * math.sin(math.radians(duty[0]))))
                        LY = int(duty[1] * math.cos(math.radians(duty[0])))
                        RX = int(duty[3] * math.sin(math.radians(duty[2])))
                        RY = int(duty[3] * math.cos(math.radians(duty[2])))
                        FR = LY - LX + RX
                        FL = LY + LX - RX
                        BL = LY - LX - RX
                        BR = LY + LX + RX
                        if duty[0] == None or duty[1]== None or duty[2] == None or duty[3] == None:
                            continue
                        self.car.motor.set_motor_model(FL, BL, FR, BR)
                    elif self.cmd_parse.command_string == self.command.CMD_CAR_ROTATE:
                        try:
                            self.car_mode = 1
                            duty = [int(self.cmd_parse.int_parameter[i]) for i in range(4)]
                            if duty[3] == 0:
                                try:
                                    stop_thread(Rotate_Mode)
                                    self.rotation_flag = False
                                except:
                                    pass
                                LX = -int((duty[1] * math.sin(math.radians(duty[0]))))
                                LY = int(duty[1] * math.cos(math.radians(duty[0])))
                                RX = int(duty[3] * math.sin(math.radians(duty[2])))
                                RY = int(duty[3] * math.cos(math.radians(duty[2])))
                                FR = LY - LX + RX
                                FL = LY + LX - RX
                                BL = LY - LX - RX
                                BR = LY + LX + RX
                                if duty[0] == None or duty[1]== None or duty[2] == None:
                                    continue
                                self.car.motor.set_motor_model(FL, BL, FR, BR)
                            elif self.rotation_flag == False:
                                try:
                                    stop_thread(self.Rotate_Mode)
                                except:
                                    pass
                                self.rotation_flag = True
                                Rotate_Mode = Thread(target=self.car.mode_rotate, args=(duty[2],))
                                Rotate_Mode.start()
                        except Exception as e:
                            print(e)
                    elif self.cmd_parse.command_string == self.command.CMD_MODE:
                        if self.cmd_parse.int_parameter[0] == 0:
                            self.car_mode = 1
                            self.car.motor.set_motor_model(0, 0, 0, 0)
                            print("Car Mode: Manual Car")
                        elif self.cmd_parse.int_parameter[0] == 1:
                            self.car_mode = 2
                            print("Car Mode: Light Car")
                        elif self.cmd_parse.int_parameter[0] == 2:
                            self.car_mode = 3
                            print("Car Mode: Infrared Car")
                        elif self.cmd_parse.int_parameter[0] == 3:
                            self.car_mode = 4
                            print("Car Mode: Ultrasonic Car")
                        elif self.cmd_parse.int_parameter[0] == 4:
                            self.car_mode = 5
                            print("Car Mode: YOLO Pedestrian Self-Driving Car")
                    
            if self.queue_cmd.empty():
                time.sleep(0.001)

    def set_threading_car_task(self, state, close_time=0.3):
        if self.car_thread is None:
            buf_state = False
        else:
            buf_state = self.car_thread.is_alive()
        if state != buf_state:
            if state:
                self.car_thread_is_running = True
                self.car_thread = threading.Thread(target=self.threading_car_task)
                self.car_thread.start()
            else:
                self.car_thread_is_running = False
                if self.car_thread is not None:
                    self.car_thread.join(close_time)
                    self.car_thread = None

    def threading_car_task(self):
        while self.car_thread_is_running:
            if self.car_mode == 1:
                pass
            elif self.car_mode == 2:
                self.car.mode_light()
                self.send_light_data()
            elif self.car_mode == 3:
                self.car.mode_infrared()
            elif self.car_mode == 4:
                self.car.mode_ultrasonic()
                self.send_sonic_data()
            elif self.car_mode == 5:
                self.mode_yolo_self_driving()
                self.send_sonic_data()
            time.sleep(0.01)

    def mode_yolo_self_driving(self, min_person_height_ratio: float = 0.25):
        """
        Autonomous self-driving task using Ultrasonic + YOLO standing/walking detections:
        - Obstacle within 30cm -> stop & turn around
        - WALKING person ahead -> stop/wait until clear
        - STANDING person ahead -> go around left or right
        - Clear path -> drive forward
        """
        dist = self.car.sonic.get_distance()
        if dist is not None and 0 < dist < 30.0:
            self.car.motor.set_motor_model(-1200, -1200, -1200, -1200)
            time.sleep(0.2)
            self.car.motor.set_motor_model(1500, 1500, -1500, -1500)
            time.sleep(0.8)
            self.car.motor.set_motor_model(0, 0, 0, 0)
            return

        relevant = [
            d for d in self.latest_detections
            if 0.15 <= d["center_x_norm"] <= 0.85 and d["height_ratio"] >= min_person_height_ratio
        ]
        walking = [d for d in relevant if d["action"] == "walking"]
        standing = [d for d in relevant if d["action"] == "standing"]

        if walking:
            self.car.motor.set_motor_model(0, 0, 0, 0)
            if self._last_yolo_action != "wait":
                print("[YOLO Self-Driving] WALKING pedestrian detected -> Waiting...")
                self._last_yolo_action = "wait"
        elif standing:
            closest = max(standing, key=lambda d: d["height_ratio"])
            if closest["center_x_norm"] >= 0.5:
                if self._last_yolo_action != "around_left":
                    print("[YOLO Self-Driving] STANDING pedestrian on right -> Going around LEFT...")
                    self._last_yolo_action = "around_left"
                self.car.motor.set_motor_model(600, 600, 2000, 2000)
            else:
                if self._last_yolo_action != "around_right":
                    print("[YOLO Self-Driving] STANDING pedestrian on left -> Going around RIGHT...")
                    self._last_yolo_action = "around_right"
                self.car.motor.set_motor_model(2000, 2000, 600, 600)
        else:
            if self._last_yolo_action != "forward":
                print("[YOLO Self-Driving] Path clear -> Driving forward...")
                self._last_yolo_action = "forward"
            self.car.motor.set_motor_model(1000, 1000, 1000, 1000)


    def set_threading_video_send(self, state, close_time=0.3):
        if self.video_thread is None:
            buf_state = False
        else:
            buf_state = self.video_thread.is_alive()
        if state != buf_state:
            if state:
                self.video_thread_is_running = True
                self.video_thread = threading.Thread(target=self.threading_video_send)
                self.video_thread.start()
            else:
                self.video_thread_is_running = False
                if self.video_thread is not None:
                    self.video_thread.join(close_time)
                    self.video_thread = None

    def threading_video_send(self):
        while self.video_thread_is_running:
            if self.tcp_server.is_video_server_connected() or self.car_mode == 5:
                self.camera.start_stream()
                while self.video_thread_is_running and (self.tcp_server.is_video_server_connected() or self.car_mode == 5):
                    frame = self.camera.get_frame()
                    if self.yolo_enabled and self.yolo_tracker is not None and self.det_model is not None and self.cls_model is not None:
                        bgr = cv2.imdecode(np.frombuffer(frame, dtype=np.uint8), cv2.IMREAD_COLOR)
                        if bgr is not None:
                            annotated, detections = self.yolo_tracker.update_and_classify_with_detections(
                                bgr, self.det_model, self.cls_model, conf=0.4, is_video_stream=True
                            )
                            self.latest_detections = detections
                            ok, enc = cv2.imencode(".jpg", annotated)
                            if ok:
                                frame = enc.tobytes()
                    if self.tcp_server.is_video_server_connected():
                        lenFrame = len(frame)
                        lengthBin = struct.pack('<I', lenFrame)
                        try:
                            self.tcp_server.send_data_to_video_client(lengthBin)
                            self.tcp_server.send_data_to_video_client(frame)
                        except:
                            break
                self.camera.stop_stream()
            else:
                time.sleep(0.1)

    def set_process_led_running(self, state, close_time=0.3):
        if self.led_process is None:
            buf_state = False
        else:
            buf_state = self.led_process.is_alive()
        if state != buf_state:
            if state:
                self.led_process_is_running = True
                if sys.platform.startswith("win"):
                    self.led_process = threading.Thread(
                        target=self.process_led_running, args=(self.queue_led,), daemon=True
                    )
                else:
                    self.led_process = multiprocessing.Process(
                        target=self.process_led_running, args=(self.queue_led,)
                    )
                self.led_process.start()
            else:
                self.led_process_is_running = False
                if self.led_process is not None:
                    try:
                        if hasattr(self.led_process, "terminate"):
                            self.led_process.terminate()
                        self.led_process.join(close_time)
                        self.led_process = None
                    except Exception as e:
                        print(f"Error terminating LED process: {e}")

    def process_led_running(self, queue_led):
        try:
            while self.led_process_is_running:
                if not queue_led.empty():
                    queue_buf_cmd = queue_led.get()
                    self.led_parse.clear_parameters()
                    self.led_parse.parse(queue_buf_cmd)
                    print("LED: {}".format(queue_buf_cmd))

                    if self.led_parse.command_string == "CMD_LED" and self.led_mode == 1:
                        try:
                            data1 = int(self.led_parse.int_parameter[0])
                            data2 = int(self.led_parse.int_parameter[1])
                            data3 = int(self.led_parse.int_parameter[2])
                            data4 = int(self.led_parse.int_parameter[3])
                            if data1==None or data2==None or data3==None or data4==None:
                                continue
                            self.led.ledIndex(data1,data2,data3,data4)
                        except:
                            pass
                    if self.led_parse.command_string == "CMD_LED_MOD":
                        if self.led_parse.int_parameter[0] == 1:
                            self.led_mode = 1
                        elif self.led_parse.int_parameter[0] == 2:
                            self.led_mode = 2
                        elif self.led_parse.int_parameter[0] == 3:
                            self.led_mode = 3
                        elif self.led_parse.int_parameter[0] == 4:
                            self.led_mode = 4
                        elif self.led_parse.int_parameter[0] == 5:
                            self.led_mode = 5
                        else:
                            self.led_mode = 0
                if self.led_mode == 1:
                    time.sleep(0.01)
                elif self.led_mode == 2:
                    self.led.following()
                elif self.led_mode == 3:
                    self.led.colorBlink(1)
                elif self.led_mode == 4:
                    self.led.rainbowbreathing()
                elif self.led_mode == 5:
                    self.led.rainbowCycle()
                elif self.led_mode == 0:
                    self.led.colorBlink(0)
                    time.sleep(0.05)
                        

        except KeyboardInterrupt:
            print("LED process interrupted, cleaning up...")
            self.led.colorBlink(0)

    def close_application(self):
        self.ui_button_state = False
        self.set_threading_cmd_receive(False)
        self.set_threading_video_send(False)
        self.set_threading_car_task(False)
        self.set_process_led_running(False)
        if self.tcp_server:
            self.tcp_server.stop_tcp_servers()
            self.tcp_server = None
        self.stop_car()
        if self.cmd_thread and self.cmd_thread.is_alive():
            self.cmd_thread.join(0.1)
        if self.video_thread and self.video_thread.is_alive():
            self.video_thread.join(0.1)
        if self.car_thread and self.car_thread.is_alive():
            self.car_thread.join(0.1)
        if self.led_process and self.led_process.is_alive():
            if hasattr(self.led_process, "terminate"):
                self.led_process.terminate()
            self.led_process.join(0.1)
        if self.app is not None:
            self.app.quit()
        sys.exit(0)

    def signal_handler(self, signal, frame):
        print("Caught Ctrl+C, stopping application...")
        self.close_application()

    def check_signals(self):
        if self.app is not None and self.app.hasPendingEvents():
            self.app.processEvents()
        if not self.ui_button_state and not self.cmd_thread_is_running and not self.video_thread_is_running and not self.led_process_is_running and not self.action_process_is_running:
            if self.app is not None:
                self.app.quit()

if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Freenove 4WD Smart Car Server')
    parser.add_argument('-t', '--terminal', action='store_true', help='Run in terminal mode (no GUI)')
    parser.add_argument('-n', '--no-gui', action='store_true', help='Run in terminal mode (no GUI)')
    parser.add_argument('--yolo', action='store_true', help='Start directly in YOLO Pedestrian Self-Driving mode')
    
    args = parser.parse_args()
    
    # Run in headless terminal mode if requested OR if PyQt5 is not installed
    headless_mode = args.terminal or args.no_gui or (not _HAS_PYQT5)
    if headless_mode:
        if not _HAS_PYQT5 and not (args.terminal or args.no_gui):
            print("PyQt5 not installed — running Server in terminal (headless) mode.")
        server_window = mywindow(headless=True)
        if args.yolo:
            server_window.car_mode = 5
            print("Car Mode: YOLO Pedestrian Self-Driving Car (active)")
        print(
            f"Server is running on {server_window.tcp_server.ip_address} "
            "(Command port: 5000, Video port: 8000). Press Ctrl+C to stop."
        )
        try:
            while True:
                time.sleep(0.5)
        except KeyboardInterrupt:
            server_window.close_application()
    else:
        # Run with GUI (existing behavior)
        myshow = mywindow(headless=False)
        if args.yolo:
            myshow.car_mode = 5
        myshow.show()
        sys.exit(myshow.app.exec_())