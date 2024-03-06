"""The code from the GitHub repository was used to run yolov5 using OpenVINO:
https://github.com/violet17/yolov5_demo
"""

from models import OpenVinoModel, YoloParams
from bbox_processing import draw_bounding_boxes, letterbox, scale_bbox_ppe, intersection_over_union
from gui import Ui_MainWindow

from timer_py import Timer
import copy
import os
import cv2
import sys
import numpy as np
from openvino.inference_engine import IECore
from typing import List, Tuple, Dict, Union

from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import  QWidget, QLabel, QApplication, QMainWindow
from PyQt5.QtCore import QThread, Qt, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap


PERSON_MODEL = 'person-detection-0200'
GREEN = (0, 255, 0)
RED = (0, 0, 255)
BLACK = (0, 0, 0)
REQUEST_ID = 0

def parse_yolo_region_ppe(
        blob: np.ndarray,
        resized_image_shape: Tuple[int, int],
        original_im_shape: Tuple[int, int],
        params: YoloParams,
        threshold: float,
        person_x1: int,
        person_y1: int
    ) -> Dict[str, Union[int, float]]:

    """
    Parses the output of the YOLO region layer for detecting personal protective equipment (PPE) in an image.

    Args:
        blob (np.ndarray): Output blob from the YOLO model.
        resized_image_shape (Tuple[int, int]): Shape of the resized image.
        original_im_shape (Tuple[int, int]): Shape of the original image.
        params (YoloParams): Parameters of the YOLO model.
        threshold (float): Threshold for confidence score to consider detection.
        person_x1 (int): X-coordinate of the left corner of the detected person.
        person_y1 (int): Y-coordinate of the left corner of the detected person.

    Returns:
        List[Dict[str, Union[int, float]]]: List of dictionaries containing information about detected objects (PPE).
            Each dictionary contains keys: 'xmin', 'xmax', 'ymin', 'ymax', 'class_id', 'confidence'.
    """

    # ------------------------------------------ Validating output parameters ------------------------------------------
    out_blob_n, out_blob_c, out_blob_h, out_blob_w = blob.shape
    predictions = 1.0 / (1.0 + np.exp(-blob))

    assert out_blob_w == out_blob_h, "Invalid size of output blob. It sould be in NCHW layout and height should " \
                                     "be equal to width. Current height = {}, current width = {}" \
                                     "".format(out_blob_h, out_blob_w)

    # ------------------------------------------ Extracting layer parameters -------------------------------------------
    orig_im_h, orig_im_w = original_im_shape
    resized_image_h, resized_image_w = resized_image_shape
    objects = list()

    # ------------------------------------------- Parsing YOLO Region output -------------------------------------------
    bbox_size = int(out_blob_c / params.num)  # 4+1+num_classes

    for row, col, n in np.ndindex(params.side, params.side, params.num):
        bbox = predictions[0, n * bbox_size:(n + 1) * bbox_size, row, col]

        x, y, width, height, object_probability = bbox[:5]
        class_probabilities = bbox[5:]
        if object_probability < threshold:
            continue
        x = (2 * x - 0.5 + col) * (resized_image_w / out_blob_w)
        y = (2 * y - 0.5 + row) * (resized_image_h / out_blob_h)
        if int(resized_image_w / out_blob_w) == 8 & int(resized_image_h / out_blob_h) == 8:  # 80x80,
            idx = 0
        elif int(resized_image_w / out_blob_w) == 16 & int(resized_image_h / out_blob_h) == 16:  # 40x40
            idx = 1
        elif int(resized_image_w / out_blob_w) == 32 & int(resized_image_h / out_blob_h) == 32:  # 20x20
            idx = 2

        width = (2 * width) ** 2 * params.anchors[idx * 6 + 2 * n]
        height = (2 * height) ** 2 * params.anchors[idx * 6 + 2 * n + 1]
        class_id = np.argmax(class_probabilities)
        confidence = object_probability

        objects.append(scale_bbox_ppe(x=x, y=y, height=height, width=width, class_id=class_id, confidence=confidence,
                                    im_h=orig_im_h, im_w=orig_im_w, person_x1 = person_x1, person_y1 = person_y1, resized_im_h=resized_image_h,
                                    resized_im_w=resized_image_w))

    return objects


class Thread(QThread):
    """
    Thread class to run inference on video frames.

    Attributes:
        finished (pyqtSignal): Signal emitted when the thread finishes.
        changePixmap (pyqtSignal): Signal emitted to update the pixmap in the UI.
    """
    finished = pyqtSignal()
    changePixmap = pyqtSignal(QImage)

    def run(self):
        """
        Run method to perform inference on video frames.
        """
        ie = IECore()
        person_net = OpenVinoModel()
        person_net.model_load(ie, PERSON_MODEL, 'CPU', True)

        model = "../yolov5n.xml"
        yolo = ie.read_network(model=model)
        yolo.batch_size = 1
        input_blob = next(iter(yolo.input_info))
        batch_size_yolo, channels_yolo, height_yolo, width_yolo = yolo.input_info[input_blob].input_data.shape
        ppe_net = ie.load_network(network=yolo, num_requests=2, device_name='CPU')

        key = -1

        video_name = os.listdir('videos')
        cap = cv2.VideoCapture('videos/' + video_name[0])

        while True:
            ret, frame = cap.read()
            if ret:

                timer = Timer('Frametime')
                timer.start()

                frame_width = int(width_yolo)
                frame_height = int(frame.shape[0] * (100 / (frame.shape[1] / width_yolo)) / 100)

                frame_size = (frame_width, frame_height)
                frame = cv2.resize(frame, frame_size)
                result_frame = copy.deepcopy(frame)

                # Detect human body and draw bounding boxes
                res = person_net.image_infer(frame)
                people = draw_bounding_boxes(result_frame, res['detection_out'][0][0]) # Отрисовка людей
                ppe_list = list()

# ------------------------------------------- PPE searching -------------------------------------------
                for obj in people:

                    # obj contains the coordinates of the corners of people in pixels of the original image
                    rgb_frame = cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB)
                    crop_frame = rgb_frame[obj['ymin']:obj['ymax'], obj['xmin']:obj['xmax']]
                    person_frame = cv2.cvtColor(crop_frame, cv2.COLOR_RGB2BGR)

                    input_person_frame = letterbox(person_frame, (width_yolo, height_yolo))

                    # Top-left corner of the detected person.
                    person_x1 = obj['xmin']
                    person_y1 = obj['ymin']
                    # Resize input_frame to network size
                    input_person_frame = input_person_frame.transpose((2, 0, 1))
                    input_person_frame = input_person_frame.reshape((batch_size_yolo, channels_yolo, height_yolo, width_yolo))

                    try:
                        ppe_net.start_async(request_id=REQUEST_ID, inputs={input_blob: input_person_frame})
                    except Exception as ex:
                        print(ex)

                    if ppe_net.requests[REQUEST_ID].wait(-1) == 0:
                        output = ppe_net.requests[REQUEST_ID].output_blobs

                        for layer_name, out_blob in output.items():
                            layer_params = YoloParams(side=out_blob.buffer.shape[2])

                            ppe_list += parse_yolo_region_ppe(
                                out_blob.buffer,
                                input_person_frame.shape[2:],
                                person_frame.shape[:-1],
                                layer_params,
                                0.5,
                                person_x1,
                                person_y1)


                        ppe_list = sorted(ppe_list, key=lambda ppe: ppe['confidence'], reverse=True)

                        for i in range(len(ppe_list)):
                            if ppe_list[i]['confidence'] == 0:
                                continue
                            for j in range(i + 1, len(ppe_list)):
                                if intersection_over_union(ppe_list[i], ppe_list[j]) > 0.4:
                                    ppe_list[j]['confidence'] = 0

                        ppe_list = [ppe for ppe in ppe_list if ppe['confidence'] >= 0.5]

                people = people + ppe_list

                elapsed = timer.elapsed(print=False)
                timer.stop()
# ------------------------------------------- Drawing ppe bboxes -------------------------------------------

                for obj in people:

                    counter = 0
                    for ppe in ppe_list:
                        if obj['class_id'] == 2 and ppe['xmin'] >= obj['xmin'] and ppe['ymin'] >= obj['ymin'] and ppe[
                            'xmax'] <= obj['xmax'] and ppe['ymax'] <= obj['ymax']:
                            counter += 1
                    if counter == 2:
                        color = GREEN
                    else:
                        color = RED

                    if obj['class_id'] != 2:
                        color = BLACK

                    cv2.rectangle(result_frame, (obj['xmin'], obj['ymin']), (obj['xmax'], obj['ymax']), color, 2)


                result_frame = cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB)
                _, _, frame_channels = result_frame.shape
                bytes_per_line = frame_channels * frame_width
                convertToQtFormat = QImage(result_frame.data, frame_width, frame_height, bytes_per_line, QImage.Format_RGB888)
                result = convertToQtFormat.scaled(1600, 900, Qt.KeepAspectRatio)
                self.changePixmap.emit(result)

                key = cv2.waitKey(1)
                if key == ord(' '):
                    while cv2.waitKey(30) != ord(' '): pass


class MainWindow(QWidget, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent=parent)
        self.setupUi(self)
        self.show()
        self.bt_start.clicked.connect(lambda: self.bt_start_click())

    def bt_start_click(self):
        th = Thread(self)
        th.changePixmap.connect(self.setImage)
        th.finished.connect(self.thread_finished)
        th.start()

    @pyqtSlot(QImage)
    def setImage(self, image):
        self.label.setPixmap(QPixmap.fromImage(image))

    def thread_finished(self):
        QApplication.quit()

    def closeEvent(self, event):
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MainWindow()
    sys.exit(app.exec_())
