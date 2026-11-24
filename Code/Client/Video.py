#!/usr/bin/python 
# -*- coding: utf-8 -*-
import numpy as np
import cv2
import socket
import io
import os
import sys
import struct
import time
from PIL import Image
from multiprocessing import Process
from Command import COMMAND as cmd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

class VideoStreaming:
    def __init__(self):
        face_xml = os.path.join(os.path.dirname(__file__), "haarcascade_frontalface_default.xml")
        self.face_cascade = (
            cv2.CascadeClassifier(face_xml)
            if hasattr(cv2, "CascadeClassifier")
            else None
        )
        self.video_Flag=True
        self.connect_Flag=False
        self.face_x=0
        self.face_y=0
        self.yolo_enabled = True
        self.auto_drive_yolo = False
        self.last_detections = []
        self._last_cmd_time = 0.0
        self._driving_state = "stopped"

        self.cls_model = None
        self.det_model = None
        self.tracker = None
        try:
            from detect_pedestrian_action import PedestrianVideoTracker, load_pedestrian_models

            self.cls_model, self.det_model, cls_path, det_path = load_pedestrian_models()
            self.tracker = PedestrianVideoTracker(history_len=10)
            print(f"Loaded YOLO standing/walking models in Client VideoStreaming ({cls_path}, {det_path}).")
        except Exception as e:
            print(f"YOLO pedestrian model not loaded in Client ({e})")
    def StartTcpClient(self,IP):
        self.client_socket1 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    def StopTcpcClient(self):
        try:
            self.client_socket.shutdown(2)
            self.client_socket1.shutdown(2)
            self.client_socket.close()
            self.client_socket1.close()
        except:
            pass

    def IsValidImage4Bytes(self,buf): 
        bValid = True
        if buf[6:10] in (b'JFIF', b'Exif'):     
            if not buf.rstrip(b'\0\r\n').endswith(b'\xff\xd9'):
                bValid = False
        else:        
            try:  
                Image.open(io.BytesIO(buf)).verify() 
            except:  
                bValid = False
        return bValid

    def face_detect(self,img):
        if sys.platform.startswith('win') or sys.platform.startswith('darwin') or sys.platform.startswith('linux'):
            # 1. Run YOLO standing/walking pedestrian detection & classification
            if self.yolo_enabled and self.tracker is not None and self.det_model is not None and self.cls_model is not None:
                img, detections = self.tracker.update_and_classify_with_detections(
                    img, self.det_model, self.cls_model, conf=0.4, is_video_stream=True
                )
                self.last_detections = detections
                if detections:
                    primary = max(detections, key=lambda d: d["height_ratio"])
                    x1, y1, x2, y2 = primary["bbox"]
                    self.face_x = float((x1 + x2) / 2.0)
                    self.face_y = float((y1 + y2) / 2.0)
                else:
                    self.face_x = 0
                    self.face_y = 0

                if self.auto_drive_yolo and self.connect_Flag:
                    self._apply_yolo_driving_decision(detections)
            elif self.face_cascade is not None:
                gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray,1.3,5)
                if len(faces)>0 :
                    for (x,y,w,h) in faces:
                        self.face_x=float(x+w/2.0)
                        self.face_y=float(y+h/2.0)
                        img= cv2.circle(img, (int(self.face_x),int(self.face_y)), int((w+h)/4), (0, 255, 0), 2)
                else:
                    self.face_x=0
                    self.face_y=0
        cv2.imwrite('video.jpg',img)

    def _apply_yolo_driving_decision(self, detections: list[dict], min_height_ratio: float = 0.25):
        """
        Translates YOLO pedestrian standing/walking detections into TCP motor commands
        when Client YOLO self-driving is enabled:
        - WALKING person ahead -> Stop / wait
        - STANDING person ahead -> Go around left or right
        - Clear path -> Drive forward
        """
        now = time.time()
        if now - self._last_cmd_time < 0.25:
            return

        relevant = [
            d for d in detections
            if 0.15 <= d["center_x_norm"] <= 0.85 and d["height_ratio"] >= min_height_ratio
        ]
        walking = [d for d in relevant if d["action"] == "walking"]
        standing = [d for d in relevant if d["action"] == "standing"]

        if walking:
            if self._driving_state != "wait":
                self.sendData(f"{cmd.CMD_MOTOR}#0#0#0#0\n")
                self._driving_state = "wait"
                self._last_cmd_time = now
        elif standing:
            closest = max(standing, key=lambda d: d["height_ratio"])
            if closest["center_x_norm"] >= 0.5:
                # Standing on right -> go around left
                self.sendData(f"{cmd.CMD_MOTOR}#600#600#2000#2000\n")
                self._driving_state = "around_left"
            else:
                # Standing on left -> go around right
                self.sendData(f"{cmd.CMD_MOTOR}#2000#2000#600#600\n")
                self._driving_state = "around_right"
            self._last_cmd_time = now
        else:
            if self._driving_state != "forward":
                self.sendData(f"{cmd.CMD_MOTOR}#1000#1000#1000#1000\n")
                self._driving_state = "forward"
                self._last_cmd_time = now
        
    def streaming(self,ip):
        stream_bytes = b' '
        try:
            self.client_socket.connect((ip, 8000))
            self.connection = self.client_socket.makefile('rb')
        except:
            #print "command port connect failed"
            pass
        while True:
            try:
                stream_bytes= self.connection.read(4) 
                leng=struct.unpack('<L', stream_bytes[:4])
                jpg=self.connection.read(leng[0])
                if self.IsValidImage4Bytes(jpg):
                            image = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                            if self.video_Flag:
                                self.face_detect(image)
                                self.video_Flag=False
            except Exception as e:
                print (e)
                break
                  
    def sendData(self,s):
        if self.connect_Flag:
            self.client_socket1.send(s.encode('utf-8'))

    def recvData(self):
        data=""
        try:
            data=self.client_socket1.recv(1024).decode('utf-8')
        except:
            pass
        return data

    def socket1_connect(self,ip):
        try:
            self.client_socket1.connect((ip, 5000))
            self.connect_Flag=True
            print ("Connection Successful !")
        except Exception as e:
            print ("Connect to server Failed!: Server IP is right? Server is opened?")
            self.connect_Flag=False

if __name__ == '__main__':
    pass

