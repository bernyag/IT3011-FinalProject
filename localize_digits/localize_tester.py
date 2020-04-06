import cv2
import numpy as np 
from matplotlib import pyplot as plt
import math
import localizer_functions as lf # custom stuff
import sys
import os

basepath = "localize_digits/errs/"

equations = lf.get_all_data_cv('/generated_images')

print(len(equations))
count = 0
err_cnt = 0
for equ_type in equations: #[equations[x] for x in equations]:
    #print(equ_type)
    for i, equ in enumerate(equations[equ_type]):
        count += 1
        if equ_type == "0div0" and i == 4:
            pass #set_trace()
        all_symbols = lf.parse_equation(equ)
        if len(all_symbols) != 3:
            err_cnt += 1
            print(f"{len(all_symbols)} is the length of all symbols, name {equ_type}/{i}")
            path = f"{basepath}{equ_type}/{i}/"
            if not os.path.exists(path): 
                os.makedirs(path)
            for j in range(0, len(all_symbols)):
                cv2.imwrite(f"{path}{j}.png", all_symbols[j])
        
print(count)
print(err_cnt)