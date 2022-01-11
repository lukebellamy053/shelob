#!/usr/bin/env python
# -*- coding: utf-8 -*-
import csv
import copy
import argparse
import itertools
from collections import Counter
from collections import deque

import cv2 as cv
import numpy as np
import mediapipe as mp


camera_instance = None


class Camera:
    
    cap_device = 0
    cap_width = 960
    cap_height = 540
    cap: cv.VideoCapture = None
    instance = None
    
    def init_camera(self):
        if self.cap is None:
            self.cap = cv.VideoCapture(self.cap_device)
            self.cap.set(cv.CAP_PROP_FRAME_WIDTH, self.cap_width)
            self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, self.cap_height)
    
    def kill_camera(self):
        self.cap.release()    
    
    
    def __init__(self, cap_device = 0, cap_width = 960, cap_height = 540):
        self.cap_device = cap_device
        self.cap_height = cap_height
        self.cap_width = cap_width
        self.init_camera()
        
    def take_picture(self):
        ret, image = self.cap.read()
        if not ret:
            return
        image = cv.flip(image, 1)  # Mirror display
        debug_image = copy.deepcopy(image)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        return ret, image, debug_image


def get_instance(cap_device = 0, cap_width = 960, cap_height = 540):
    if Camera.instance is None:
        Camera.instance = Camera(cap_device, cap_width, cap_height)
    return Camera.instance