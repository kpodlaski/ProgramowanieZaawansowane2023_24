import json
import numpy as np

from ColorSpaceOperation import ColorSpaceOperation
from ImageFilterOperation import ImageFilterOperation
from ImageInitOperation import ImageInitOperation
from ImageOperation import ImageOperation


class PipelineBuilder(ImageOperation):
    def __init__(self, jsonfile):
        self.operator = None
        self.parseJson(jsonfile)


    def parseJson(self, jsonfile):
        with open(jsonfile) as user_file:
            file_contents = user_file.read()
            conf = json.loads(file_contents)
        opId = 1
        for op in reversed(conf):
            if op['type'] == 'filter':
                kernel = np.array(op['filter'], dtype=np.double)
                kernel /= kernel.sum()
                self.operator = ImageFilterOperation('op_'+str(opId), kernel, op['times'], next=self.operator)
            else:
                self.operator = ColorSpaceOperation('op_'+str(opId), op['type'], next=self.operator)
            opId+=1
        self.operator = ImageInitOperation('op_0', '../imgs', next=self.operator)

    def doTask(self):
        self.operator.doTask()


    def implemented_operation(self, image_file):
        pass
