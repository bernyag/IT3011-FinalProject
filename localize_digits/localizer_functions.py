import cv2
import numpy as np 
from matplotlib import pyplot as plt
import math

def extract_number(stats, src, i):
    left = stats[i, cv2.CC_STAT_LEFT]
    top = stats[i, cv2.CC_STAT_TOP]
    width = stats[i, cv2.CC_STAT_WIDTH]
    height = stats[i, cv2.CC_STAT_HEIGHT]
    number = src[top:top + height, left:left + width]
    cv2.imwrite(str(i) + ".png", number)
    return number

def make_square_out_of(number):
    height, width, _ = number.shape
    left, right, top, bottom = (0,0,0,0) #initialize borders
    if height > width:
        # add border left and right
        strech = (height - width) / 2
        # if strech is not an integer, strech 1 pixel more to the right
        left = math.floor(strech)
        right = math.ceil(strech)
    elif height < width:
        strech = (width - height) / 2
        top = math.floor(strech)
        bottom = math.ceil(strech)

    COLOR_WHITE = (255,255,255)
    # this function does all the magic
    dst = cv2.copyMakeBorder(number, top, bottom, left, right, cv2.BORDER_CONSTANT, None, COLOR_WHITE)
    return dst

def extract_square_number(stats, src, i):
    number = extract_number(stats, src, i)
    return make_square_out_of(number)

def extract_square_division_symbol(pair, stats, src):
    i, overlap = pair
    left = stats[i + 1, cv2.CC_STAT_LEFT]
    top = stats[i + 1, cv2.CC_STAT_TOP]
    right = stats[i + 1, cv2.CC_STAT_WIDTH] + left
    bottom = stats[i + 1, cv2.CC_STAT_HEIGHT] + top
    for j in overlap:
        new_left = stats[j + 1, cv2.CC_STAT_LEFT]
        new_top = stats[j + 1, cv2.CC_STAT_TOP]
        left = min(new_left, left)
        top = min(new_top, top)
        right = max(stats[j + 1, cv2.CC_STAT_WIDTH] + new_left, right)
        bottom = max(stats[j + 1, cv2.CC_STAT_HEIGHT] + new_top, bottom)
    division = src[top:bottom, left:right]
    return make_square_out_of(division) 