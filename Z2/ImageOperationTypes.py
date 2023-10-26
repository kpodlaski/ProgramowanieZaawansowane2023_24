from enum import Enum
import cv2

class ImageOperationTypes(Enum):
    HSV_BGR = 1,
    BGR_HSV = 2,
    FILTER = 3


