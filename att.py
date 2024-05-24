import cv2
import numpy as np
import matplotlib.pyplot as plt

def hdr_to_brightness(hdr_image):
    R, G, B = cv2.split(hdr_image)
    brightness = 0.265 * R + 0.670 * G + 0.065 * B
    return brightness

def log10_delta_La(log10_La):
    if log10_La < -3.94:
        return -3.81
    elif -3.94 <= log10_La < -1.44:
        return (0.405 * log10_La + 1.6)**2.18 - 3.81
    elif -1.44 <= log10_La < -0.0184:
        return log10_La - 1.345
    elif -0.0184 <= log10_La < 1.9:
        return (0.249 * log10_La + 0.65)**2.7 - 1.67
    else:  # log10_La >= 1.9
        return log10_La - 2.205

def get_jnd(La):
    log10_La = np.log10(La)
    result = log10_delta_La(log10_La)
    return 10**result

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

def show_histogram(histogram):
    plt.bar(range(len(bins_centers) - 1), histogram, width=1.0)
    plt.xlabel('Bin Index')
    plt.ylabel('Frequency')
    plt.title('Histogram of Brightness Values')
    plt.show()

def show_histograms(*histograms, labels=None):
    num_bins = len(histograms[0])
    bin_indices = np.arange(num_bins)
    width = 0.8 / len(histograms)  # 设置条形宽度

    # 绘制每个直方图
    for i, histogram in enumerate(histograms):
        if labels is not None:
            label = labels[i]
        else:
            label = f'Histogram {i+1}'
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



# 加载图片
hdr_image = cv2.imread('data/input_images/input_images/input_hdr/cadik-desk02_mid.hdr', cv2.IMREAD_ANYDEPTH)


# 将颜色通道顺序修改为 RGB
hdr_image = cv2.cvtColor(hdr_image, cv2.COLOR_BGR2RGB)

brightness_channel = hdr_to_brightness(hdr_image)

max_brightness = np.max(brightness_channel)
min_brightness = np.min(brightness_channel)
if min_brightness == 0:
    min_brightness = 0.0001

print(f"Max brightness: {max_brightness}")
print(f"Min brightness: {min_brightness}")

# 我们的bins_centers是bins的边界，和文章中的不同，我们这样做是为了简单，因为文章中只提到了bins的中心，没有提到bins的边界是怎么划分的
n = 15  # 公式bi+1 = bi + n * delta_b中的n
bins_centers = [min_brightness]

while bins_centers[-1] < max_brightness:
    current_bin_center = bins_centers[-1]
    delta_b = get_jnd(current_bin_center)
    next_bin_center = current_bin_center + n * delta_b
    bins_centers.append(next_bin_center)

if bins_centers[-1] > max_brightness:
    bins_centers.pop()

print(f"Number of bins: {len(bins_centers) - 1}")

# 每十个 bin 打印成一行
for i in range(0, len(bins_centers), 10):
    print("Bins:", end=" ")
    for j in range(i, min(i + 10, len(bins_centers))):
        print(f"{bins_centers[j]:.2f}", end=" ")
    print()


f_histogram = calculate_histogram_2D(brightness_channel, bins_centers)
# 归一化第一个直方图
f_histogram = normalize_histogram(f_histogram)
# show_histogram(f_histogram)

print(f"Histogram bins: {f_histogram}")

# Sort brightness values
sorted_brightness = np.sort(brightness_channel.flatten())
print(len(sorted_brightness))
# Initialize new brightness levels and counts
new_brightness_levels = []
new_brightness_counts = []

# Start with the first brightness level
current_brightness = sorted_brightness[0]
current_count = 1

# Iterate through the sorted brightness values
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

print(len(new_brightness_levels))
r_histogram = calculate_histogram_1D(new_brightness_levels, bins_centers)
r_histogram = normalize_histogram(r_histogram)
print(f"Histogram bins: {r_histogram}")

w = 0.8
c_histogram = w * f_histogram + (1 - w) * r_histogram
print(c_histogram)

# # 绘制三个直方图
# show_histograms(f_histogram, r_histogram, c_histogram, labels=['Original', 'Refined', 'Combined'])
cumulative_histogram = np.cumsum(c_histogram)
cumulative_histogram = np.insert(cumulative_histogram, 0, 0)


LUT = create_LUT(bins_centers, cumulative_histogram)
print("LUT:")
print(LUT)


ldr_brightness_channel = map_hdr_brightness_to_ldr(brightness_channel, LUT)

s = 0.67
R, G, B = cv2.split(hdr_image)
ldr_image_R = (R/brightness_channel)**s * ldr_brightness_channel
ldr_image_G = (G/brightness_channel)**s * ldr_brightness_channel
ldr_image_B = (B/brightness_channel)**s * ldr_brightness_channel

ldr_image = cv2.merge((ldr_image_R, ldr_image_G, ldr_image_B))

# 将像素值限制在 0 到 255 之间，并转换为 uint8 类型
ldr_image = np.clip(ldr_image, 0, 255).astype(np.uint8)



plt.imshow(ldr_image)
plt.title('LDR Image (Mapped)')
plt.axis('off')

plt.show()
bgr_ldr_image = cv2.cvtColor(ldr_image, cv2.COLOR_RGB2BGR)
cv2.imwrite("mapped_ldr.jpg", bgr_ldr_image)
