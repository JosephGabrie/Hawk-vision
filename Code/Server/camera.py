import time
try:
    from picamera2 import Picamera2, Preview
    from picamera2.encoders import H264Encoder, JpegEncoder
    from picamera2.outputs import FileOutput
    from libcamera import Transform
    _HAS_PICAMERA2 = True
except ImportError:
    _HAS_PICAMERA2 = False
from threading import Condition, Thread
import io
import sys
import cv2
import numpy as np

class StreamingOutput(io.BufferedIOBase):
    def __init__(self):
        """Initialize the StreamingOutput class."""
        self.frame = None
        self.condition = Condition()  # Initialize the condition variable for thread synchronization

    def write(self, buf: bytes) -> int:
        """Write a buffer to the frame and notify all waiting threads."""
        with self.condition:
            self.frame = buf             # Update the frame buffer with new data
            self.condition.notify_all()  # Notify all waiting threads that new data is available
        return len(buf)

class Camera:
    def __init__(self, preview_size: tuple = (640, 480), hflip: bool = False, vflip: bool = False, stream_size: tuple = (400, 300)):
        """Initialize the Camera class (Picamera2 on Raspberry Pi, OpenCV fallback on Windows/Linux)."""
        self.stream_size = stream_size
        self.hflip = hflip
        self.vflip = vflip
        self.streaming_output = StreamingOutput()
        self.streaming = False
        self._cv_cap = None
        self._cv_thread = None

        if _HAS_PICAMERA2:
            self.camera = Picamera2()
            self.transform = Transform(hflip=1 if hflip else 0, vflip=1 if vflip else 0)
            preview_config = self.camera.create_preview_configuration(main={"size": preview_size}, transform=self.transform)
            self.camera.configure(preview_config)
            self.stream_config = self.camera.create_video_configuration(main={"size": stream_size}, transform=self.transform)
        else:
            self.camera = None

    def start_image(self) -> None:
        """Start the camera preview and capture."""
        if self.camera is not None:
            self.camera.start_preview(Preview.QTGL)
            self.camera.start()

    def save_image(self, filename: str) -> dict:
        """Capture and save an image to the specified file."""
        try:
            if self.camera is not None:
                return self.camera.capture_file(filename)
            cap = cv2.VideoCapture(0)
            ret, frame = cap.read()
            cap.release()
            if ret:
                cv2.imwrite(filename, frame)
                return {"file": filename}
            return None
        except Exception as e:
            print(f"Error capturing image: {e}")
            return None

    def _cv_stream_loop(self) -> None:
        while self.streaming and self._cv_cap is not None:
            ret, frame = self._cv_cap.read()
            if not ret or frame is None:
                # Generate a synthetic placeholder frame if no webcam is attached
                frame = np.zeros((self.stream_size[1], self.stream_size[0], 3), dtype=np.uint8)
                cv2.putText(
                    frame,
                    "No Camera Detected",
                    (20, self.stream_size[1] // 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                )
                time.sleep(0.1)
            else:
                frame = cv2.resize(frame, self.stream_size)
            ok, enc = cv2.imencode(".jpg", frame)
            if ok:
                self.streaming_output.write(enc.tobytes())
            time.sleep(0.03)

    def start_stream(self, filename: str = None) -> None:
        """Start the video stream or recording."""
        if not self.streaming:
            if self.camera is not None:
                if self.camera.started:
                    self.camera.stop()
                self.camera.configure(self.stream_config)
                if filename:
                    encoder = H264Encoder()
                    output = FileOutput(filename)
                else:
                    encoder = JpegEncoder()
                    output = FileOutput(self.streaming_output)
                self.camera.start_recording(encoder, output)
                self.streaming = True
            else:
                self._cv_cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) if sys.platform.startswith("win") else cv2.VideoCapture(0)
                if not self._cv_cap.isOpened():
                    self._cv_cap = cv2.VideoCapture(0)
                self.streaming = True
                self._cv_thread = Thread(target=self._cv_stream_loop, daemon=True)
                self._cv_thread.start()

    def stop_stream(self) -> None:
        """Stop the video stream or recording."""
        if self.streaming:
            try:
                self.streaming = False
                if self.camera is not None:
                    self.camera.stop_recording()
                elif self._cv_cap is not None:
                    if self._cv_thread is not None:
                        self._cv_thread.join(0.5)
                    self._cv_cap.release()
                    self._cv_cap = None
            except Exception as e:
                print(f"Error stopping stream: {e}")

    def get_frame(self) -> bytes:
        """Get the current frame from the streaming output."""
        with self.streaming_output.condition:
            self.streaming_output.condition.wait()
            return self.streaming_output.frame

    def save_video(self, filename: str, duration: int = 10) -> None:
        """Save a video for the specified duration."""
        self.start_stream(filename)
        time.sleep(duration)
        self.stop_stream()

    def close(self) -> None:
        """Close the camera."""
        if self.streaming:
            self.stop_stream()
        if self.camera is not None:
            self.camera.close()

if __name__ == '__main__':
    print('Program is starting ... ')                    # Print a message indicating the start of the program
    camera = Camera()                                    # Create a Camera instance

    print("View image...")
    camera.start_image()                                 # Start the camera preview
    time.sleep(10)                                       # Wait for 10 seconds
    
    print("Capture image...")
    camera.save_image(filename="image.jpg")              # Capture and save an image
    time.sleep(1)                                        # Wait for 1 second

    '''
    print("Stream video...")
    camera.start_stream()                                # Start the video stream
    time.sleep(3)                                        # Stream for 3 seconds
    
    print("Stop video...")
    camera.stop_stream()                                 # Stop the video stream
    time.sleep(1)                                        # Wait for 1 second

    print("Save video...")
    camera.save_video("video.h264", duration=3)          # Save a video for 3 seconds
    time.sleep(1)                                        # Wait for 1 second
    
    print("Close camera...")
    camera.close()                                       # Close the camera
    '''