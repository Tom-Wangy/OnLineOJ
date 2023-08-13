import sys
# import hyperparameter
from ACGA import ACGA_Algorithm
import json
import numpy as np
import utils
import pretreatment
from hyperparameter import args
from pretreatment import Data_loader



# data = Data_loader(1,args)
def run(number):
    # data = Data_loader(number,args)
    acga = ACGA_Algorithm(1)
    result = acga.Acga_Run(1)

    return result    
