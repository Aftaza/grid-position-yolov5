import cv2
import time
import sys
import numpy as np

class YOLOv5Detector:
    INPUT_WIDTH = 640
    INPUT_HEIGHT = 640
    SCORE_THRESHOLD = 0.2
    NMS_THRESHOLD = 0.4
    CONFIDENCE_THRESHOLD = 0.4
    COLORS = [(255, 255, 0), (0, 255, 0), (0, 255, 255), (255, 0, 0)]

    def __init__(self, model_path, class_path, is_cuda=False):
        self.net = self.build_model(model_path, is_cuda)
        self.capture = self.load_capture()
        self.class_list = self.load_classes(class_path)
        self.frame_count = 0
        self.total_frames = 0
        self.fps = -1
        self.start_time = time.time_ns()

    def build_model(self, model_path, is_cuda):
        net = cv2.dnn.readNet(model_path)
        if is_cuda:
            print("Attempting to use CUDA")
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
        else:
            print("Running on CPU")
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        return net

    def load_capture(self):
        return cv2.VideoCapture(0)

    def load_classes(self, class_path):
        with open(class_path, "r") as f:
            class_list = [cname.strip() for cname in f.readlines()]
        return class_list

    def detect(self, image):
        blob = cv2.dnn.blobFromImage(image, 1/255.0, (self.INPUT_WIDTH, self.INPUT_HEIGHT), swapRB=True, crop=False)
        self.net.setInput(blob)
        preds = self.net.forward()
        return preds

    def wrap_detection(self, input_image, output_data):
        class_ids = []
        confidences = []
        boxes = []

        rows = output_data.shape[0]
        image_width, image_height, _ = input_image.shape
        x_factor = image_width / self.INPUT_WIDTH
        y_factor = image_height / self.INPUT_HEIGHT

        for r in range(rows):
            row = output_data[r]
            confidence = row[4]
            if confidence >= self.CONFIDENCE_THRESHOLD:
                classes_scores = row[5:]
                _, _, _, max_indx = cv2.minMaxLoc(classes_scores)
                class_id = max_indx[1]
                if (classes_scores[class_id] > 0.25):
                    confidences.append(confidence)
                    class_ids.append(class_id)

                    x, y, w, h = row[0].item(), row[1].item(), row[2].item(), row[3].item()
                    left = int((x - 0.5 * w) * x_factor)
                    top = int((y - 0.5 * h) * y_factor)
                    width = int(w * x_factor)
                    height = int(h * y_factor)
                    box = np.array([left, top, width, height])
                    boxes.append(box)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.25, 0.45)
        result_class_ids = [class_ids[i] for i in indexes]
        result_confidences = [confidences[i] for i in indexes]
        result_boxes = [boxes[i] for i in indexes]
        return result_class_ids, result_confidences, result_boxes

    def format_yolov5(self, frame):
        row, col, _ = frame.shape
        _max = max(col, row)
        result = np.zeros((_max, _max, 3), np.uint8)
        result[0:row, 0:col] = frame
        return result

    def run(self):
        while True:
            ret, frame = self.capture.read()
            if frame is None:
                print("End of stream")
                break

            input_image = self.format_yolov5(frame)
            outs = self.detect(input_image)
            class_ids, confidences, boxes = self.wrap_detection(input_image, outs[0])

            self.frame_count += 1
            self.total_frames += 1

            for (classid, confidence, box) in zip(class_ids, confidences, boxes):
                color = self.COLORS[int(classid) % len(self.COLORS)]
                cv2.rectangle(frame, box, color, 2)
                cv2.rectangle(frame, (box[0], box[1] - 20), (box[0] + box[2], box[1]), color, -1)
                cv2.putText(frame, self.class_list[classid], (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

            if self.frame_count >= 30:
                end_time = time.time_ns()
                self.fps = 1000000000 * self.frame_count / (end_time - self.start_time)
                self.frame_count = 0
                self.start_time = time.time_ns()

            if self.fps > 0:
                fps_label = f"FPS: {self.fps:.2f}"
                cv2.putText(frame, fps_label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.imshow("output", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Finished by user")
                break

        print("Total frames:", self.total_frames)
        self.capture.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    is_cuda = len(sys.argv) > 1 and sys.argv[1] == "cuda"
    detector = YOLOv5Detector("conf/yolov5n.onnx", "conf/class.txt", is_cuda)
    detector.run()
