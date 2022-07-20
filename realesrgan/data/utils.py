import random
import cv2
import math
import numpy as np
import torch

def get_components_bbox(lm):
    item_dict = {}
    map_left_eye = list(range(36, 42))
    map_right_eye = list(range(42, 48))
    # map_mouth = list(range(48, 68))

    mean_left_eye = np.mean(lm[map_left_eye], 0)  # (x, y)
    half_len_left_eye = np.max((np.max(np.max(lm[map_left_eye], 0) - np.min(lm[map_left_eye], 0)) / 2, 16))
    item_dict['left_eye'] = [mean_left_eye[0], mean_left_eye[1], half_len_left_eye]
    # mean_left_eye[0] = 512 - mean_left_eye[0]  # for testing flip
    half_len_left_eye *= 1.4

    # eye_right
    mean_right_eye = np.mean(lm[map_right_eye], 0)
    half_len_right_eye = np.max((np.max(np.max(lm[map_right_eye], 0) - np.min(lm[map_right_eye], 0)) / 2, 16))
    item_dict['right_eye'] = [mean_right_eye[0], mean_right_eye[1], half_len_right_eye]
    # mean_right_eye[0] = 512 - mean_right_eye[0]  # # for testing flip
    half_len_right_eye *= 1.4

    # mouth
    # mean_mouth = np.mean(lm[map_mouth], 0)
    # half_len_mouth = np.max((np.max(np.max(lm[map_mouth], 0) - np.min(lm[map_mouth], 0)) / 2, 16))
    # item_dict['mouth'] = [mean_mouth[0], mean_mouth[1], half_len_mouth]

    return item_dict


def get_component_coordinates(lmks):
    """Get facial component (left_eye, right_eye, mouth) coordinates from a pre-loaded pth file"""

    components_bbox = get_components_bbox(lmks)

    locations = []
    for part in ['left_eye', 'right_eye']:
        mean = components_bbox[part][0:2]
        half_len = components_bbox[part][2]
        loc = np.hstack((mean - half_len + 1, mean + half_len))
        loc = torch.from_numpy(loc).float()
        locations.append(loc)
    return locations

