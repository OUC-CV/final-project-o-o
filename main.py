import cv2
import numpy as np
import matplotlib.pyplot as plt
from wls import *
from att import *
from twt import *


hdr_image = cv2.imread('data/input_images/input_images/input_hdr/cadik-desk02_mid.hdr', cv2.IMREAD_ANYDEPTH)
att_aloghtrom(hdr_image, "att_test.jpg")


