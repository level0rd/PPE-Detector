"""The code from the GitHub repository was used to run yolov5 using OpenVINO:
https://github.com/violet17/yolov5_demo
"""
from __future__ import print_function, division

import copy
import os
import cv2
import imutils
import sys
from video import Ui_MainWindow
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import  QWidget, QLabel, QApplication, QMainWindow
from PyQt5.QtCore import QThread, Qt, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap
from PIL import Image

import numpy
import qimage2ndarray
import numpy as np
from openvino.inference_engine import IECore

PERSON_MODEL = 'person-detection-0200'
GREEN = (0, 255, 0)
RED = (0, 0, 255)
BLACK = (0, 0, 0)

class MainWindow(QWidget, Ui_MainWindow):

    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent=parent)
        self.setupUi(self)
        self.show()
        self.bt_start.clicked.connect(lambda: self.bt_start_click())

    def bt_start_click(self):
        th = Thread(self)
        th.changePixmap.connect(self.setImage)
        th.start()

    @pyqtSlot(QImage)
    def setImage(self, image):
        self.label.setPixmap(QPixmap.fromImage(image))

    def closeEvent(self, event):
        event.accept()
	    

class openvino_model:
    def __init__(self):
        pass

    def model_load(self, ie, model_name, device='CPU', verbose=False):
        model_tmpl = '../' + model_name +'.xml'
        self.net    = ie.read_network(model_tmpl.format(model_name, 'xml'))
        self.exenet = ie.load_network(self.net, device)
        self.iblob_name  = list(self.exenet.input_info)
        self.iblob_shape = [ self.exenet.input_info[n].tensor_desc.dims for n in self.iblob_name]
        self.oblob_name  = list(self.exenet.outputs)
        self.oblob_shape = [ self.exenet.outputs[n].shape for n in self.oblob_name]
        self.device = device
        if verbose:
            print(model_name, self.iblob_name, self.iblob_shape, self.oblob_name, self.oblob_shape)

    def image_infer(self, *args):
        inputs = {}
        for img, bname, bshape in zip(args, self.iblob_name, self.iblob_shape):
            n,c,h,w = bshape
            img = cv2.resize(img, (w,h))
            #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # OMZ models expect BGR image input
            img = img.transpose((2,0,1))
            img = img.reshape((n,c,h,w))
            inputs[bname] = img
        self.res = self.exenet.infer(inputs)
        return self.res

def draw_bounding_boxes(img, bboxes, threshold=0.75, color=(255,255,255), thickness=2): # 0.80
    h, w, c = img.shape
    objects = list()
    for bbox in bboxes:
        id, label, conf, xmin, ymin, xmax, ymax = bbox
        if conf > threshold:
            x1 = max(0, int(xmin * w))
            y1 = max(0, int(ymin * h))
            x2 = max(0, int(xmax * w))
            y2 = max(0, int(ymax * h))
            objects.append(dict(xmin=x1, xmax=x2, ymin=y1, ymax=y2, class_id=2, confidence=conf))
    return objects

class YoloParams:
    # ------------------------------------------- Extracting layer parameters ------------------------------------------
    # Magic numbers are copied from yolo samples
    def __init__(self, side):
        self.num = 3  # if 'num' not in param else int(param['num'])
        self.coords = 4  # if 'coords' not in param else int(param['coords'])
        self.classes = 80  # if 'classes' not in param else int(param['classes'])
        self.side = side
        self.anchors = [10.0, 13.0, 16.0, 30.0, 33.0, 23.0, 30.0, 61.0, 62.0, 45.0, 59.0, 119.0, 116.0, 90.0, 156.0,
                        198.0,
                        373.0, 326.0]  # if 'anchors' not in param else [float(a) for a in param['anchors'].split(',')]


def letterbox(img, size=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]
    w, h = size

    # Scale ratio (new / old)
    r = min(h / shape[0], w / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = w - new_unpad[0], h - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 64), np.mod(dh, 64)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (w, h)

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border

    top2, bottom2, left2, right2 = 0, 0, 0, 0

    if img.shape[0] != h:
        top2 = (h - img.shape[0]) // 2
        bottom2 = top2
        img = cv2.copyMakeBorder(img, top2, bottom2, left2, right2, cv2.BORDER_CONSTANT, value=color)  # add border
    elif img.shape[1] != w:
        left2 = (w - img.shape[1]) // 2
        right2 = left2
        img = cv2.copyMakeBorder(img, top2, bottom2, left2, right2, cv2.BORDER_CONSTANT, value=color)  # add border

    return img

def scale_bbox_ppe(x, y, height, width, class_id, confidence, im_h, im_w, person_x1, person_y1, resized_im_h=640, resized_im_w=640):
    gain = min(resized_im_w / im_w, resized_im_h / im_h)  # gain  = old / new
    pad = (resized_im_w - im_w * gain) / 2, (resized_im_h - im_h * gain) / 2  # wh padding
    x = int((x - pad[0]) / gain)
    y = int((y - pad[1]) / gain)

    w = int(width / gain)
    h = int(height / gain)

    xmin = max(0, int(x - w / 2))
    ymin = max(0, int(y - h / 2))
    xmax = min(im_w, int(xmin + w))
    ymax = min(im_h, int(ymin + h))

    # Obtaining PPE on the source image through the pixels x1/y1 coordinates of a person in pixels
    fullx1 = person_x1 + xmin
    fully1 = person_y1 + ymin
    fullx2 = person_x1 + xmax
    fully2 = person_y1 + ymax

    # Method item() used here to convert NumPy types to native types for compatibility with functions, which don't
    # support Numpy types (e.g., cv2.rectangle doesn't support int64 in color parameter)
    return dict(xmin=fullx1, xmax=fullx2, ymin=fully1, ymax=fully2, class_id=class_id.item(), confidence=confidence.item())

def entry_index(side, coord, classes, location, entry):
    side_power_2 = side ** 2
    n = location // side_power_2
    loc = location % side_power_2
    return int(side_power_2 * (n * (coord + classes + 1) + entry) + loc)


def parse_yolo_region_ppe(blob, resized_image_shape, original_im_shape, params, threshold, person_x1, person_y1):
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


def intersection_over_union(box_1, box_2):
    width_of_overlap_area = min(box_1['xmax'], box_2['xmax']) - max(box_1['xmin'], box_2['xmin'])
    height_of_overlap_area = min(box_1['ymax'], box_2['ymax']) - max(box_1['ymin'], box_2['ymin'])
    if width_of_overlap_area < 0 or height_of_overlap_area < 0:
        area_of_overlap = 0
    else:
        area_of_overlap = width_of_overlap_area * height_of_overlap_area
    box_1_area = (box_1['ymax'] - box_1['ymin']) * (box_1['xmax'] - box_1['xmin'])
    box_2_area = (box_2['ymax'] - box_2['ymin']) * (box_2['xmax'] - box_2['xmin'])
    area_of_union = box_1_area + box_2_area - area_of_overlap
    if area_of_union == 0:
        return 0
    return area_of_overlap / area_of_union

class Thread(QThread):
    changePixmap = pyqtSignal(QImage)

    def run(self):
        ie = IECore()
        person_det = openvino_model()
        person_det.model_load(ie, PERSON_MODEL, 'CPU', True)

        model = "../yolov5n.xml"
        net = ie.read_network(model=model)
        # assert len(net.input_info.keys()) == 1, "Sample supports only YOLO V3 based single input topologies"
        input_blob = next(iter(net.input_info))
        net.batch_size = 1
        # Read and pre-process input images
        n, c, h, w = net.input_info[input_blob].input_data.shape
        exec_net = ie.load_network(network=net, num_requests=2, device_name='CPU')

        key = -1

        video_name = os.listdir('videos') 
        cap = cv2.VideoCapture('videos/' + video_name[0])

        current_request_id = 0
        next_request_id = 1

        while True:
            ret, input_img = cap.read()
            if ret:
                width = int(640)
                height = int(input_img.shape[0] * (100 / (input_img.shape[1] / 640)) / 100)

                frame_size = (width, height)

                # Resize image
                input_img = cv2.resize(input_img, frame_size)
                res_frame = copy.deepcopy(input_img)

                # Detect human body and draw bounding boxes
                people_coordinates = person_det.image_infer(input_img)
                people = draw_bounding_boxes(res_frame, people_coordinates['detection_out'][0][0]) 
                ppes = list()

                for obj in people:

		    
                    rgb_image = cv2.cvtColor(res_frame, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(rgb_image)

                    # Cropping a person from an image
		    crop_frame = pil_img.crop(
		    # The coordinates of the corners of the bounding boxes of people in pixels of the source image
					     (obj['xmin'], obj['ymin'], obj['xmax'], obj['ymax']))  
                    person_frame_rgb = numpy.array(crop_frame)
                    person_frame = cv2.cvtColor(person_frame_rgb, cv2.COLOR_RGB2BGR)

                    request_id = current_request_id
                    h = 640

                    input_person_frame = letterbox(person_frame, (w, h))

 		    # Left corner of the found person
                    person_x1 = obj['xmin']
                    person_y1 = obj['ymin']

                    # Resize input_frame to network size
		    # Change data layout from HWC to CHW
                    input_person_frame = input_person_frame.transpose((2, 0, 1))  
                    input_person_frame = input_person_frame.reshape((n, c, h, w))
			
                    try:
                    exec_net.start_async(request_id=request_id, inputs={input_blob: input_person_frame})
                    except Exception as ex:
                        print(ex)

                    if exec_net.requests[current_request_id].wait(-1) == 0:
                        output = exec_net.requests[current_request_id].output_blobs
                        for layer_name, out_blob in output.items():
                            layer_params = YoloParams(side=out_blob.buffer.shape[2])
                            ppes += parse_yolo_region_ppe(out_blob.buffer, input_person_frame.shape[2:],
                                                          person_frame.shape[:-1], layer_params,
                                                          0.5, person_x1, person_y1)

                        ppes = sorted(ppes, key=lambda ppe: ppe['confidence'], reverse=True)
			    
                        for i in range(len(ppes)):
                            if ppes[i]['confidence'] == 0:
                                continue
                            for j in range(i + 1, len(ppes)):
                                if intersection_over_union(ppes[i], ppes[j]) > 0.4:
                                    ppes[j]['confidence'] = 0

                        ppes = [ppe for ppe in ppes if ppe['confidence'] >= 0.5]


                people = people + ppes

                for obj in people:
                    # Define color of bboxes
                    counter = 0
                    for ppe in ppes:
                        if obj['class_id'] == 2 and ppe['xmin'] >= obj['xmin'] and ppe['ymin'] >= obj['ymin'] and ppe[
                            'xmax'] <= obj['xmax'] and ppe['ymax'] <= obj['ymax']:
                            counter += 1
                    if counter == 2:
                        color = GREEN
                    else:
                        color = RED

                    if obj['class_id'] != 2:
                        color = BLACK

                    cv2.rectangle(res_frame, (obj['xmin'], obj['ymin']), (obj['xmax'], obj['ymax']), color, 2)

                rgb_image = cv2.cvtColor(res_frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytesPerLine = ch * w
                convertToQtFormat = QImage(rgb_image.data, w, h, bytesPerLine, QImage.Format_RGB888)
                result_image = convertToQtFormat.scaled(1600, 900, Qt.KeepAspectRatio) # 640, 480
                self.changePixmap.emit(result_image)

                key = cv2.waitKey(1)
                if key == ord(' '):
                    while cv2.waitKey(30) != ord(' '): pass


if __name__ == '__main__':
    # Resolving command-line arguments for an application
    app = QApplication(sys.argv)
    ex = MainWindow()
    sys.exit(app.exec_())
