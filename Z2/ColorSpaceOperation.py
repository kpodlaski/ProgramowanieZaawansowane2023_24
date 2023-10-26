from ImageOperation import ImageOperation
import cv2 as cv

string_mapping = {
    'rgb-hsv' : cv.COLOR_RGB2HSV,
    'bgr-hsv' : cv.COLOR_BGR2HSV,
    'hsv-rgb' : cv.COLOR_HSV2RGB,
    'hsv-bgr' : cv.COLOR_HSV2BGR,
}
class ColorSpaceOperation(ImageOperation):
    def __init__(self, task_name, str_desc, next=None):
        ImageOperation.__init__(self,task_name+"_"+str_desc+"_",next=next)
        splitter_index =str_desc.find("-")
        self.initialSpace = str_desc[:splitter_index].upper()
        self.finalSpace = str_desc[splitter_index+1:].upper()
        self._colorConversion = string_mapping[str_desc.lower()]

    def implemented_operation(self, img):
        print("Color conversion")
        print(self._colorConversion)
        result = cv.cvtColor(img, self._colorConversion)
        return result

