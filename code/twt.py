import cv2
import matplotlib.pyplot as plt
import numpy as np
from att import luminance_compression
from wlsFilter import wlsFilter

def cal_histogram(gray_image, save_path):
    min_val = np.min(gray_image)
    max_val = np.max(gray_image)
    print(f"min_val: {min_val} max_val: {max_val}")

    # 将图像平展为一维数组
    pixel_values = gray_image.flatten()

    # 绘制直方图
    plt.figure(figsize=(10, 5))
    plt.hist(pixel_values, bins=256, range=(min_val, max_val), color='gray', alpha=0.75)
    plt.title("Pixel Value Distribution")
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")
    plt.xlim([min_val, max_val])
    plt.savefig(save_path)
    plt.close()

def hdr_to_brightness_twt(brightness):
    """
    转变为灰度图，放大，归一化，返回图片
    :param hdr_image:
    :return: ans
    """
    # 亮度值扩大 10^6 倍
    brightness = np.log(brightness * 10**6 + 1)
    ret = brightness/np.max(brightness)
    return ret

def method1(B_base, B_detail, u):
    D_detail = 2 * np.arctan(B_detail * u) / np.pi
    cal_histogram(D_detail, "outputs/twt/histogram_D_detail.png")
    plt.imsave("outputs/twt/D_detail.png", D_detail, cmap='gray')
    B_base = B_base + D_detail

    cal_histogram(B_base, "outputs/twt/histogram_B_base_merged.png")
    plt.imsave("outputs/twt/B_base_merged.png", B_base, cmap='gray')

    # 反对数化
    B_base_ = 10 ** (10 * B_base - 4)  # 3.11
    # twt算法的前半部分
    D_base = luminance_compression(B_base_, is_att=False)
    plt.imsave("outputs/twt/D_base.png", D_base, cmap='gray')

    return D_base

def method2(B_base, B_detail, u):
    D_detail = u * np.arctan(B_detail * 30)
    cal_histogram(D_detail, "outputs/twt/histogram_D_detail.png")
    plt.imsave("outputs/twt/D_detail.png", D_detail, cmap='gray')

    # 反对数化
    B_base_ = 10 ** (10 * B_base - 4)  # 3.11
    # twt算法的前半部分
    D_base = luminance_compression(B_base_, is_att=False)
    plt.imsave("outputs/twt/D_base.png", D_base, cmap='gray')

    return D_base + D_detail



def twt_algorithm(hdr_image, save_path, s, u,adjust_s, adjust_u, apply_method1):
    # 前期预处理
    B, G, R = cv2.split(hdr_image)
    brightness_channel = 0.265 * R + 0.670 * G + 0.065 * B

    normlized_brightness_channel = hdr_to_brightness_twt(brightness_channel)

    if adjust_s == False:
        # 尺度分解处理
        # 使用WLS滤波器
        # print(B_base.shape)
        if adjust_u == False:
            B_base = wlsFilter(normlized_brightness_channel)#此处调用Matlab程序
            B_base = np.array(B_base)

            cal_histogram(B_base, "outputs/twt/histogram_B_base.png")
            plt.imsave("outputs/twt/B_base.png", B_base, cmap='gray')

            B_detail = normlized_brightness_channel - B_base  # 3.10

            cal_histogram(B_detail, "outputs/twt/histogram_B_detail.png")
            plt.imsave("outputs/twt/B_detail.png", B_detail, cmap='gray')

            np.save("B_base.npy", B_base)
            np.save("B_detail.npy", B_detail)


        else:
            B_base = np.load("B_base.npy")
            B_detail = np.load("B_detail.npy")


        if apply_method1 == True:
            image_merged = method1(B_base, B_detail, u)
        else:
            image_merged = method2(B_base, B_detail, u)
        np.save("twt_data.npy", image_merged)
    else:
        image_merged = np.load("twt_data.npy")

    #颜色校正，此处有可调试参数s
    B, G, R = cv2.split(hdr_image)
    ldr_image_B = (B/brightness_channel)**s * image_merged
    ldr_image_G = (G/brightness_channel)**s * image_merged
    ldr_image_R = (R/brightness_channel)**s * image_merged
    ldr_image = cv2.merge((ldr_image_B, ldr_image_G, ldr_image_R))


    # 将像素值限制在 0 到 255 之间，并转换为 uint8 类型
    ldr_image = np.clip(ldr_image, 0, 255).astype(np.uint8)

    image_rgb = cv2.cvtColor(ldr_image, cv2.COLOR_BGR2RGB)

    # 显示RGB图像
    cal_histogram(ldr_image_B, "outputs/twt/histogram_final.png")

    cv2.imwrite(save_path, ldr_image)