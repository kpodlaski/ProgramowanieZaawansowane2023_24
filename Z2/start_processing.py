from ImageInitOperation import ImageInitOperation
from ColorSpaceOperation import ColorSpaceOperation
import numpy as np
import os


from ImageFilterOperation import ImageFilterOperation

##TEST CASE
# operator = None
# operator = ColorSpaceOperation('test','hsv-bgr')
# kernel = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])
# operator = ImageFilterOperation('test',kernel, 5, next = operator)
# operator = ColorSpaceOperation('test','bgr-hsv', next = operator)
# operator = ImageInitOperation('test', '../imgs', next=operator)
# operator.doTask()

from PiplelineBuilder import PipelineBuilder
operator = PipelineBuilder('operations.json')
operator.doTask()