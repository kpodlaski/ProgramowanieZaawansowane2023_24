import numpy as np
import cv2
class ImageOperation:
    def __init__(self, task_name, next=None):
        self.next = next
        self.times = 1
        self.output_folder = "../out"
        self.working_folder = "../work"
        self.input_folder = "../in"
        self.task_name = task_name
    def operation(self, img):
        for i in range(self.times):
            result = self.implemented_operation(img)
            self.save_result(result, i)
        if self.next is not None:
            self.next.operation(result)
        else:
            self.save_result(result,0,final=True)
            cv2.imshow("final image",result)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    ## To be defined in children classes
    def implemented_operation(self, img):
        pass
    def save_result(self, result, i, final=False):
        #save to a file results with unique name
        if not final:
            cv2.imwrite("../work/"+self.task_name+"_"+str(i)+".png",result)
        else:
            cv2.imwrite("../out/"+self.task_name + ".png", np.array(result))
