import matlab.engine
import cv2
import numpy as np
import matplotlib.pyplot as plt
from att import *
from twt import *
from evaluation import *
import os

# 定义函数来生成输出路径
def generate_output_paths(input_path, output_dir_att, output_dir_twt):
    filename = os.path.basename(input_path)
    base_name, _ = os.path.splitext(filename)
    output_path_att = os.path.join(output_dir_att, f"{base_name}.png")
    output_path_twt = os.path.join(output_dir_twt, f"{base_name}.png")
    return output_path_att, output_path_twt

input_path = 'data/input_images/input_images/input_hdr/doll.hdr'
output_dir_att = 'outputs/att'
output_dir_twt = 'outputs/twt'

# 生成输出路径
output_path_att, output_path_twt = generate_output_paths(input_path, output_dir_att, output_dir_twt)

# 读取HDR图像
hdr_image = cv2.imread(input_path, cv2.IMREAD_ANYDEPTH)

# 调用算法
#att_algorithm(hdr_image, output_path_att, s=0.67, adjust_s=False)
twt_algorithm(hdr_image, output_path_twt, s=0.6, u=50, adjust_s=False, adjust_u=True, apply_method1=False)