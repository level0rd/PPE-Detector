import cv2
import numpy as np
from typing import List, Tuple, Dict, Union


def draw_bounding_boxes(
        img: np.ndarray,
        bboxes: List[Tuple[float, float, float, float, float, float, float]],
        threshold: float = 0.75,
        color: Tuple[int, int, int] = (255, 255, 255),
        thickness: int = 2
) -> List[Dict[str, float]]:
    """
    Draw bounding boxes on an image based on the provided bounding box coordinates.

    Args:
        img (np.ndarray): Input image array.
        bboxes (List[Tuple[float, float, float, float, float, float, float]]): List of bounding boxes,
            each represented by a tuple (id, label, confidence, xmin, ymin, xmax, ymax).
        threshold (float): Confidence threshold for considering a bounding box.
        color (Tuple[int, int, int]): Bounding box color in BGR format.
        thickness (int): Thickness of the bounding box lines.

    Returns:
        List[Dict[str, float]]: List of dictionaries representing the drawn bounding boxes with keys
            'xmin', 'xmax', 'ymin', 'ymax', 'class_id', and 'confidence'.
    """
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

def letterbox(
        img: np.ndarray,
        size: Tuple[int, int] = (640, 640),
        color: Tuple[int, int, int] = (114, 114, 114),
        auto: bool = True,
        scaleFill: bool = False,
        scaleup: bool = True
    ) -> np.ndarray:
    """
    Resize image to fit a specified size while maintaining aspect ratio, adding padding if necessary.

    Args:
        img (np.ndarray): Input image array.
        size (Tuple[int, int]): Target size (width, height) of the output image.
        color (Tuple[int, int, int]): Padding color in BGR format.
        auto (bool): Whether to use minimum rectangle.
        scaleFill (bool): Whether to stretch the image.
        scaleup (bool): Whether to scale up the image.

    Returns:
        np.ndarray: Resized image with padding.
"""
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

def scale_bbox_ppe(
        x: int,
        y: int,
        height: int,
        width: int,
        class_id: int,
        confidence: float,
        im_h: int,
        im_w: int,
        person_x1: int,
        person_y1: int,
        resized_im_h:
        int = 640,
        resized_im_w: int = 640
    ) -> Dict[str, Union[int, float]]:
    """
    Scale bounding box coordinates and dimensions from resized image to original image size and calculate
    coordinates of the bounding box relative to the original image.

    Args:
        x (int): x-coordinate of the bounding box center in the resized image.
        y (int): y-coordinate of the bounding box center in the resized image.
        height (int): Height of the bounding box in the resized image.
        width (int): Width of the bounding box in the resized image.
        class_id (int): Class ID of the detected object.
        confidence (float): Confidence score of the detected object.
        im_h (int): Height of the original image.
        im_w (int): Width of the original image.
        person_x1 (int): x-coordinate of the top-left corner of the detected person bounding box in pixels.
        person_y1 (int): y-coordinate of the top-left corner of the detected person bounding box in pixels.
        resized_im_h (int, optional): Height of the resized image. Defaults to 640.
        resized_im_w (int, optional): Width of the resized image. Defaults to 640.

    Returns:
        Dict[str, Union[int, float]]: Dictionary containing scaled bounding box coordinates and other information.
    """
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

    fullx1 = person_x1 + xmin
    fully1 = person_y1 + ymin
    fullx2 = person_x1 + xmax
    fully2 = person_y1 + ymax

    # Method item() used here to convert NumPy types to native types for compatibility with functions, which don't
    # support Numpy types (e.g., cv2.rectangle doesn't support int64 in color parameter)
    return dict(xmin=fullx1, xmax=fullx2, ymin=fully1, ymax=fully2, class_id=class_id.item(), confidence=confidence.item())


def intersection_over_union(box_1: Dict[str, Union[int, float]], box_2: Dict[str, Union[int, float]]) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.

    Args:
        box_1 (Dict[str, Union[int, float]]): Dictionary containing coordinates of the first bounding box.
        box_2 (Dict[str, Union[int, float]]): Dictionary containing coordinates of the second bounding box.

    Returns:
        float: Intersection over Union (IoU) score.
    """
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
