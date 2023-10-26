from ImageOperation import ImageOperation
import cv2

class ImageFilterOperation(ImageOperation):
    def __init__(self,task_name,filter, times = 1, next=None):
        ImageOperation.__init__(self,task_name+"filering")
        self.kernel = filter
        self.times = times

    def implemented_operation(self, img):
        print("filtering image")
        result = cv2.filter2D(img, ddepth=-1, kernel = self.kernel)
        return result