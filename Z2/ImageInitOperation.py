from ImageOperation import ImageOperation
import os
import cv2
import json


class ImageInitOperation(ImageOperation):
    def __init__(self,  task_name,path, next):
        ImageOperation.__init__(self,task_name+"_initialization")
        self.next = next
        self.file_names = []
        print(path)
        print(os.path.isdir(path))
        print(os.listdir(path))
        if os.path.isdir(path):
            self.file_names = [os.path.join(path,f) for f in os.listdir(path)]
        elif os.path.isfile(path):
            self.file_names.append(path)
        else:
            print(self.file_names)
            raise RuntimeError("Bad path parameter, it have to be file_path or directory")

    def doTask(self):
        print("preprocessing", self.file_names)
        for file in self.file_names:
            self.operation(file)


    def implemented_operation(self, image_file):
        print("Reading file", image_file)
        img =  cv2.imread(image_file)
        cv2.imshow("Initial image", img)
        return img

