import cv2
import matplotlib.pyplot as plt
import numpy as np

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



def att_algorithm(hdr_image, save_path, s, adjust_s):
    """
    ATT算法的实现，
    """
    # 将hdr图像转为亮度图
    brightness_channel = hdr_to_brightness_att(hdr_image)
    if adjust_s == False:
    # 亮度图压缩
        ldr_brightness_channel = luminance_compression(brightness_channel)
        np.save("brightness_channel.npy", ldr_brightness_channel)
    else:
        ldr_brightness_channel = np.load("brightness_channel.npy")


    B, G, R = cv2.split(hdr_image)
    ldr_image_B = (B / brightness_channel) ** s * ldr_brightness_channel
    ldr_image_G = (G / brightness_channel) ** s * ldr_brightness_channel
    ldr_image_R = (R / brightness_channel) ** s * ldr_brightness_channel
    ldr_image = cv2.merge((ldr_image_B, ldr_image_G, ldr_image_R))

    cal_histogram(ldr_image_B, "outputs/att/histogtam_final.png")
    # 将像素值限制在 0 到 255 之间，并转换为 uint8 类型
    ldr_image = np.clip(ldr_image, 0, 255).astype(np.uint8)
    # ldr_image = color_correctness(hdr_image, brightness_channel, ldr_brightness_channel)
    cv2.imwrite(save_path, ldr_image)


def color_correctness(hdr_image, brightness_channel, ldr_brightness_channel):
    # 颜色校正，此处有可调试参数s
    # TODO:实现检查映射情况的代码,以便调试参数s
    B, G, R = cv2.split(hdr_image)
    s = 0.67
    learning_rate = 0.1
    threshold = 0.01
    stop_condition = False
    max_iterations = 1000
    iteration = 0

    while not stop_condition and iteration < max_iterations:
        # 使用当前 s 值计算映射图像
        ldr_image_B = (B / brightness_channel) ** s * ldr_brightness_channel
        ldr_image_G = (G / brightness_channel) ** s * ldr_brightness_channel
        ldr_image_R = (R / brightness_channel) ** s * ldr_brightness_channel

        # 计算超过 255 的像素比例和最大像素值
        B_over_ratio = np.sum(ldr_image_B > 255) / ldr_image_B.size
        G_over_ratio = np.sum(ldr_image_G > 255) / ldr_image_G.size
        R_over_ratio = np.sum(ldr_image_R > 255) / ldr_image_R.size

        # 根据超过 255 的像素比例和最大像素值调整 s 值
        s -= learning_rate * (B_over_ratio + G_over_ratio + R_over_ratio) / 3

        # 更新迭代次数
        iteration += 1
        print(f"iteration: {iteration} s: {s}")

        # 检查停止条件
        if B_over_ratio < threshold and G_over_ratio < threshold and R_over_ratio < threshold:
            stop_condition = True

    ldr_image = cv2.merge((ldr_image_B, ldr_image_G, ldr_image_R))
    # 将像素值限制在 0 到 255 之间，并转换为 uint8 类型
    ldr_image = np.clip(ldr_image, 0, 255).astype(np.uint8)
    return ldr_image


def dynamically_choose_n(min_brightness, max_brightness, n=10):
    """
    实现了动态的选择合适的n,使得直方图的直方数目在[70, 90]内
    :return:bins_centers
    """
    stepsize = 0.1
    within_target_range = False

    while not within_target_range:
        bins_centers = [min_brightness]
        while bins_centers[-1] < max_brightness:
            current_bin_center = bins_centers[-1]
            delta_b = get_jnd(current_bin_center)
            next_bin_center = current_bin_center + n * delta_b
            bins_centers.append(next_bin_center)

        if bins_centers[-1] > max_brightness:
            bins_centers.pop()

        length = len(bins_centers) - 1
        print(n, length)
        if length <= 90 and length >= 70:
            return bins_centers
        elif length > 90:
            n = n + stepsize
        else:
            n = n - stepsize


def luminance_compression(brightness_channel, w=0.8, is_att=True):
    """
    亮度图压缩，将亮度图的数值范围压缩/拓展至[0 - 255],att算法使用此函数可将较大的数值范围压缩，twt算法使用此函数可将较小的数值范围放大
    """
    max_brightness = np.max(brightness_channel)
    min_brightness = np.min(brightness_channel)
    if min_brightness == 0:
        min_brightness = 0.0000001

    print(f"Max brightness: {max_brightness}")
    print(f"Min brightness: {min_brightness}")

    # 我们的bins_centers是bins的边界，和文章中的不同，我们这样做是为了简单，因为文章中只提到了bins的中心，没有提到bins的边界是怎么划分的

    bins_centers = dynamically_choose_n(min_brightness, max_brightness)
    print(f"num of bins:{len(bins_centers) - 1}")
    # 每十个 bin 打印成一行

    the_first_hist_over255 = 0
    for i in range(0, len(bins_centers), 10):
        print("Bins:", end=" ")
        for j in range(i, min(i + 10, len(bins_centers))):
            if the_first_hist_over255 ==0 and bins_centers[j] > 255:
                the_first_hist_over255 = j
            print(f"{bins_centers[j]:.5f}", end=" ")
        print()

    f_histogram = calculate_histogram_2D(brightness_channel, bins_centers)
    # 归一化第一个直方图
    f_histogram = normalize_histogram(f_histogram)

    # 此处加了一个直方图的意义，可以直观地看到一幅hdr图像像素值在[0-255]和(255, bigger]区间内的大致分布情况
    if is_att == False:
        save_path = "outputs/twt/brightness_channel_distribution.png"
    else: save_path = "outputs/att/brightness_channel_distribution.png"

    show_histogram(f_histogram, the_first_hist_over255, save_path)

    # print(f"Histogram bins: {f_histogram}")

    # 对亮度进行排序
    sorted_brightness = np.sort(brightness_channel.flatten())
    print(len(sorted_brightness))
    # 存放亮度值差异较明显的点
    new_brightness_levels = []
    new_brightness_counts = []

    current_brightness = sorted_brightness[0]
    current_count = 1

    for i in range(1, len(sorted_brightness)):
        next_brightness = sorted_brightness[i]
        delta_b = get_jnd(current_brightness)

        if next_brightness - current_brightness <= delta_b:
            # If the brightness difference is within JND, merge pixels
            current_count += 1
        else:
            # Otherwise, finalize the current bin and start a new one
            new_brightness_levels.append(current_brightness)
            new_brightness_counts.append(current_count)
            current_brightness = next_brightness
            current_count = 1

    new_brightness_levels.append(current_brightness)
    new_brightness_counts.append(current_count)

    print(len(new_brightness_levels))  # 经过遍历后得到的亮度差异较明显的点的数量
    r_histogram = calculate_histogram_1D(new_brightness_levels, bins_centers)
    r_histogram = normalize_histogram(r_histogram)
    #print(f"Histogram bins: {r_histogram}")

    c_histogram = w * f_histogram + (1 - w) * r_histogram
    #print(c_histogram)

    # # 绘制三个直方图
    # show_histograms(f_histogram, r_histogram, c_histogram, labels=['Original', 'Refined', 'Combined'])
    cumulative_histogram = np.cumsum(c_histogram)
    cumulative_histogram = np.insert(cumulative_histogram, 0, 0)

    LUT = create_LUT(bins_centers, cumulative_histogram)
    print("LUT:")
    print(LUT)

    ldr_brightness_channel = map_hdr_brightness_to_ldr(brightness_channel, LUT)

    return ldr_brightness_channel


def hdr_to_brightness_att(hdr_image):
    B, G, R = cv2.split(hdr_image)
    brightness = 0.265 * R + 0.670 * G + 0.065 * B
    return brightness


def log10_delta_La(log10_La):
    if log10_La < -3.94:
        return -3.81
    elif -3.94 <= log10_La < -1.44:
        return (0.405 * log10_La + 1.6) ** 2.18 - 3.81
    elif -1.44 <= log10_La < -0.0184:
        return log10_La - 1.345
    elif -0.0184 <= log10_La < 1.9:
        return (0.249 * log10_La + 0.65) ** 2.7 - 1.67
    else:  # log10_La >= 1.9
        return log10_La - 2.205


def get_jnd(La):
    log10_La = np.log10(La)
    result = log10_delta_La(log10_La)
    return 10 ** result


def calculate_histogram_2D(brightness_channel, bins_centers):
    histogram = np.zeros(len(bins_centers) - 1)

    for pixel in np.nditer(brightness_channel):
        for i in range(len(bins_centers) - 1):
            if bins_centers[i] <= pixel < bins_centers[i + 1]:
                histogram[i] += 1
                break

    return histogram


def calculate_histogram_1D(brightness, bins_centers):
    histogram = np.zeros(len(bins_centers) - 1)

    for pixel in brightness:
        for i in range(len(bins_centers) - 1):
            if bins_centers[i] <= pixel < bins_centers[i + 1]:
                histogram[i] += 1
                break

    return histogram


def normalize_histogram(histogram):
    total = np.sum(histogram)
    normalized_histogram = histogram / total
    return normalized_histogram


def show_histogram(histogram,  the_first_hist_over255, save_path):
    plt.bar(range(len(histogram)), histogram, width=1.0)
    if the_first_hist_over255 is not None:
        plt.axvline(x=the_first_hist_over255, color='r', linestyle='--', label='Pixels Over 255')
    plt.xlabel('Bin Index')
    plt.ylabel('Frequency')
    plt.title('Histogram of Brightness Values')
    plt.savefig(save_path)
    plt.close()


def show_histograms(*histograms, labels = None):
    num_bins = len(histograms[0])
    bin_indices = np.arange(num_bins)
    width = 0.8 / len(histograms)  # 设置条形宽度

    # 绘制每个直方图
    for i, histogram in enumerate(histograms):
        if labels is not None:
            label = labels[i]
        else:
            label = f'Histogram {i + 1}'
        plt.bar(bin_indices + i * width, histogram, width=width, label=label)

    plt.xlabel('Bin Index')
    plt.ylabel('Frequency')
    plt.title('Histograms of Brightness Values')
    plt.legend()
    plt.show()


def create_LUT(bins_centers, cumulative_histogram):
    LUT = np.zeros((len(bins_centers), 2))
    LUT[:, 0] = bins_centers
    LUT[:, 1] = cumulative_histogram * 255
    return LUT


def map_hdr_brightness_to_ldr(brightness_channel, LUT):
    hdr_values = LUT[:, 0]
    ldr_values = LUT[:, 1]

    ldr_brightness_channel = np.interp(brightness_channel.flatten(), hdr_values, ldr_values)
    ldr_brightness_channel = ldr_brightness_channel.reshape(brightness_channel.shape)
    return ldr_brightness_channel
