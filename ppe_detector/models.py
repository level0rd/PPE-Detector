import cv2
import numpy as np


class OpenVinoModel:
    """
    Class for loading and running inference with OpenVINO models.
    """
    def __init__(self):
        pass

    def model_load(self, ie, model_name, device='CPU', verbose=False):
        """
        Load OpenVINO model.

        Args:
            ie: InferenceEngine object.
            model_name (str): Name of the model.
            device (str): Device to load the model on (default is 'CPU').
            verbose (bool): Whether to print verbose information (default is False).
        """
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
        """
        Perform inference on input images.

        Args:
            *args: Input images to perform inference on.

        Returns:
            dict: Dictionary containing inference results.
        """
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


class YoloParams:
    """
    YOLO Parameters class for extracting layer parameters.
    Magic numbers are copied from YOLO samples.
    """
    # ------------------------------------------- Extracting layer parameters ------------------------------------------
    # Magic numbers are copied from yolo samples
    def __init__(self, side):
        """
               Initialize YoloParams object.

               Args:
                   side (int): The side parameter for YOLO.

               Attributes:
                   num (int): The number of predictions.
                   coords (int): The number of coordinates.
                   classes (int): The number of classes.
                   side (int): The side parameter for YOLO.
                   anchors (list): The list of anchor values.
               """
        self.num = 3  # if 'num' not in param else int(param['num'])
        self.coords = 4  # if 'coords' not in param else int(param['coords'])
        self.classes = 80  # if 'classes' not in param else int(param['classes'])
        self.side = side
        self.anchors = [10.0, 13.0, 16.0, 30.0, 33.0, 23.0, 30.0, 61.0, 62.0, 45.0, 59.0, 119.0, 116.0, 90.0, 156.0,
                        198.0,
                        373.0, 326.0]  # if 'anchors' not in param else [float(a) for a in param['anchors'].split(',')]
