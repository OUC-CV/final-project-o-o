import cv2
import matplotlib.pyplot as plt
import numpy as np
from att import luminance_compression
from wls import wlsFilter

def hdr_to_brightness_twt(hdr_image):
    """
    转变为灰度图，放大，归一化，返回图片
    :param hdr_image:
    :return: ans
    """
    B, G, R = cv2.split(hdr_image)
    brightness = (R + G + B)/3
    # 亮度值扩大 10^6 倍
    brightness = np.log(brightness * 10**6 + 1)
    ret = brightness/np.max(brightness)
    return ret

def twt_aloghtrom(hdr_image, save_path, u = 30):
    # 前期预处理
    brightness_channel = hdr_to_brightness_twt(hdr_image)

    # 尺度分解处理
    # 使用WLS滤波器
    B_base = wlsFilter(brightness_channel) #TODO： 此处接入matlab程序
    B_detail = hdr_image - B_base #3.10

    # 反对数化
    B_base_ = 10**(10*B_base - 4) #3.11
    # twt算法的前半部分
    D_base = luminance_compression(B_base_)

    # 细节层 进行细节增强
    u = 30
    D_detail = 2 * np.arctan(B_detail * u) / np.pi

    image_merged = D_detail + D_base

    #颜色校正，此处有可调试参数s
    B, G, R = cv2.split(hdr_image)
    s = 0.6
    ldr_image_B = (B/brightness_channel)**s * D_base
    ldr_image_G = (G/brightness_channel)**s * D_base
    ldr_image_R = (R/brightness_channel)**s * D_base
    ldr_image = cv2.merge((ldr_image_B, ldr_image_G, ldr_image_R))

    # 将像素值限制在 0 到 255 之间，并转换为 uint8 类型
    ldr_image = np.clip(ldr_image, 0, 255).astype(np.uint8)

    cv2.imwrite(save_path, ldr_image)
