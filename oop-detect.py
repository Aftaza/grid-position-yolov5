import cv2
import time
import numpy as np

class FaceGridDetector:
    def __init__(self, face_cascade_path, grid_shape=(3, 3)):
        self.face_cascade = cv2.CascadeClassifier(face_cascade_path)
        self.grid_shape = grid_shape
        self.cap = cv2.VideoCapture(0)
        self.prev_frame_time = 0
        self.new_frame_time = 0
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.center = {'x': 0, 'y': 0}
        self.gridbox = {'x': [], 'y': []}

    def draw_grid(self, img, color=(0, 255, 255), thickness=2):
        h, w, _ = img.shape
        rows, cols = self.grid_shape
        dy, dx = h / rows, w / cols

        # Clear previous grid lines
        self.gridbox['x'].clear()
        self.gridbox['y'].clear()

        # draw vertical lines
        for x in np.linspace(start=dx, stop=w-dx, num=cols-1):
            x = int(round(x))
            self.gridbox['x'].append(x)
            cv2.line(img, (x, 0), (x, h), color=color, thickness=thickness)

        # draw horizontal lines
        for y in np.linspace(start=dy, stop=h-dy, num=rows-1):
            y = int(round(y))
            self.gridbox['y'].append(y)
            cv2.line(img, (0, y), (w, y), color=color, thickness=thickness)

        return img

    def identify_grid_position(self, center, faces_detected):
        if len(faces_detected) == 0:
            return (-1, -1)
        x_pos = sum(center['x'] > x for x in self.gridbox['x'])
        y_pos = sum(center['y'] > y for y in self.gridbox['y'])
        return (x_pos, y_pos)

    def run(self):
        while True:
            ret, img = self.cap.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            self.new_frame_time = time.time()

            if len(faces) > 0:
                for (x, y, w, h) in faces:
                    # To draw a rectangle in a face
                    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 2)
                    # draw center box
                    self.center = {'x': int(x + w / 2), 'y': int(y + h / 2)}
                    cv2.circle(img, (self.center['x'], self.center['y']), 5, (0, 0, 255), -1)
            else:
                self.center = {'x': -1, 'y': -1}

            fps = 1 / (self.new_frame_time - self.prev_frame_time)
            self.prev_frame_time = self.new_frame_time

            fps = int(fps)
            fps = f"{fps} fps"

            img = self.draw_grid(img)
            grid_position = self.identify_grid_position(self.center, faces)

            cv2.putText(img, fps, (7, 30), self.font, 1, (100, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(img, f"Grid Position: {grid_position}", (7, 60), self.font, 1, (100, 255, 0), 2, cv2.LINE_AA)

            # Display an image in a window
            cv2.imshow('img', img)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Close the window
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    face_cascade_path = 'haarcascade_frontalface_default.xml'
    detector = FaceGridDetector(face_cascade_path)
    detector.run()
