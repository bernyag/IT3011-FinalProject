import cv2
import os
import numpy as np 
from matplotlib import pyplot as plt
import math
import sys


def get_all_data_cv(fName):
    directory=f'{os.getcwd()}/'+fName
    labels = {} 
    for entry in os.scandir(directory):
        label = entry.name      # The operand / operator. Example: '2'
        images = []             

        for picture in os.scandir(entry):
            image = cv2.imread(picture.path)
            images.append(image) 
        labels[label] = images  
    print('Test')  
    return labels

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


### more advanced FUNCTIONS


def get_x_coord_generator(stats):
    return lambda img_pair : stats[img_pair[0], cv2.CC_STAT_LEFT]


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

def remove_overlap_duplicates(division_symbols_pairs):
    # it might look like this: (0, [4]), (3, [0,4]) in which case we only want to keep one
    # the one which has more overlapping parts
    for ele_i, ele_overlaps in division_symbols_pairs:
        # remove item
        needs_removal = False
        for _, other_overlaps in division_symbols_pairs:
            # conditions for a bigger overlap: our element is part of other overlap
            # all the elements that overlap with us are also part of the other overlap
            # as the overlap is ordered left to right our ele_overlaps cannot contain the other_i
            if (ele_i in other_overlaps and 
                all(elem in other_overlaps for elem in ele_overlaps)):
                needs_removal = True
                break
        if needs_removal:
            division_symbols_pairs.remove((ele_i, ele_overlaps))

    return division_symbols_pairs


def split_division_rest(num_labels, stats):
    overlaps = calculate_overlaps(num_labels, stats)
    #print(overlaps)
    # the overlap array looks e.g. like this: [(0, []), (1, []), (2, [1]), (3, [])]
    # this means 1 overlaps with 2, i.e. 1 starts after 2
    # so we need to filter out all those which have a length bigger than one and handle them
    division_symbols_pairs = list(filter(lambda x: len(x[1]) > 0, overlaps))
    
    division_symbols_pairs = remove_overlap_duplicates(division_symbols_pairs)

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
        image = extract_square_division_symbol(pair, stats, src)
        i, _ = pair
        division_symbols.append((i,image))
    return division_symbols
#division_symbolsS

# this function combines all the work done in the localizer part. 
# An openCV image is being fed into it and an array of 
# openCV images with the individual characters is returned
def parse_equation(equ):
    # convert images to grayscale etc
    num_labels, _, stats, _ = prepare_image(equ)
    # split division symbols and rest into two indice pairs
    division_symbols_pairs, digit_indices = split_division_rest(num_labels, stats)
    # get the digits
    digits = []
    for i in digit_indices:
        number = extract_number(stats, equ, i)
        squared = make_square_out_of(number)
        digits.append((i - 1, squared))
    # concatenate all symbols
    all_symbols = digits + get_division_symbols(division_symbols_pairs, stats, equ)
    # sort the symbols by the x coordinate, leading to a correctly sorted array
    get_x_coord = get_x_coord_generator(stats) #lambda x: stats[x[0], cv2.CC_STAT_LEFT]
    sorted_symbols = sorted(all_symbols, key = get_x_coord)
    return [img for i, img in sorted_symbols]

path='../generated_images'
#symbols=parse_image(path)