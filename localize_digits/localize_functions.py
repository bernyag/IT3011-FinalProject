#!/usr/bin/env python
# coding: utf-8


import cv2
import numpy as np 
from matplotlib import pyplot as plt
import math
import localizer_functions # custom stuff
import sys


def get_x_coord(img_pair):
    i, _ = img_pair
    return stats[i, cv2.CC_STAT_LEFT]


# this function converts an opencv image to grayscale and runs the connectedComponents analyisis on it
# 
# ### Parameters:
# 
# input:
# src - an opencv image
# 
# output:
# (num_labels, labels, stats, centroids) - returns what connectedComponents returns


def prepare_image(src):
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    thresh = cv2.bitwise_not(thresh)
    connectivity = 4  # You need to choose 4 or 8 for connectivity type
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh , connectivity , cv2.CV_32S, cv2.CCL_DEFAULT)
    return (num_labels, labels, stats, centroids)


# this function splits the labels into division symbols and everything else
# 
# this is done by analyzing which labels are on top of each other and classifying three of those as a division symbol. The corresponding labels are removed from the digit_indices and an entry in division_symbol pairs is generated. 
# a pair consists of (a, list), where a is the index of the line and list is a list of the indices of the three elements of a division symbol

# In[4]:


def is_in_other_component(overlaps, i):
    #any(len(elem) == 5 for elem in listOfStrings)
    return any(any(x == i for x in pair[1]) for pair in overlaps)


def calculate_overlaps(num_labels, stats):
    overlaps = [(i - 1,[]) for i in range(1, num_labels)] # zero based
    # we want to iterate from left to right
    positions = np.arange(1, num_labels)
    coord = lambda x: stats[x, cv2.CC_STAT_LEFT]
    sorted_positions = sorted(positions, key=coord)
    #print(f"numlabel={num_labels}, sort positions={sorted_positions}")
    for i in range(0, num_labels - 1):
        # do not add duplicates
        if is_in_other_component(overlaps, i):
            continue
        first_index = sorted_positions[i]
        left_i = stats[first_index, cv2.CC_STAT_LEFT]
        width_i = stats[first_index, cv2.CC_STAT_WIDTH]
        for j in range(i + 1, num_labels - 1):
            #print(f"i={i}, j={j}, first_index={first_index}")
            second_index = sorted_positions[j]
            left_j = stats[second_index, cv2.CC_STAT_LEFT]
            # if left of j is within limits of i, add to the overlap list
            if left_j in range(left_i, left_i + width_i):
                _, l = overlaps[first_index - 1]
                l.append(second_index - 1)
    return overlaps


def split_division_rest(num_labels, stats):
    overlaps = calculate_overlaps(num_labels, stats)
    #print(overlaps)
    # the overlap array looks e.g. like this: [(0, []), (1, []), (2, [1]), (3, [])]
    # this means 1 overlaps with 2, i.e. 1 starts after 2
    # so we need to filter out all those which have a length bigger than one and handle them
    division_symbols_pairs = list(filter(lambda x: len(x[1]) > 0, overlaps))
    digit_indices = np.arange(1, num_labels)
    # now filter all division symbols from our normal array, we will handle them later
    for e in division_symbols_pairs:
        i, list_of_division_operator = e
        #print(f"list to delete indices:{list_of_division_operator}, list={digit_indices}")
        digit_indices = np.delete(digit_indices, list_of_division_operator + [i])
    #print(f"division symbols={division_symbols_pairs}, digits={digit_indices}")
    return (division_symbols_pairs, digit_indices)


# this function extracts all the division symbols from the pairs using stats and the src image
def get_division_symbols(division_symbols_pairs, stats, src):
    division_symbols = []
    for pair in division_symbols_pairs:
        image = localizer_functions.extract_square_division_symbol(pair, stats, src)
        i, _ = pair
        division_symbols.append((i,image))
    return division_symbols
#division_symbolsS