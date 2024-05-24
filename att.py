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

    # Assign pixels to bins
    for pixel in np.nditer(brightness_channel):
        for i in range(len(bins_centers) - 1):
            if bins_centers[i] <= pixel < bins_centers[i + 1]:
                histogram[i] += 1
                break

    return histogram

def calculate_histogram_1D(brightness, bins_centers):
    histogram = np.zeros(len(bins_centers) - 1)

    # Assign pixels to bins
    for pixel in brightness:
        for i in range(len(bins_centers) - 1):
            if bins_centers[i] <= pixel < bins_centers[i + 1]:
                histogram[i] += 1
                break

    return histogram

# Load HDR image
hdr_image = cv2.imread('data/input_images/input_images/input_hdr/AtriumMorning.hdr', cv2.IMREAD_ANYDEPTH)

brightness_channel = hdr_to_brightness(hdr_image)

max_brightness = np.max(brightness_channel)
min_brightness = np.min(brightness_channel)
if min_brightness == 0:
    min_brightness = 0.0001

print(f"Max brightness: {max_brightness}")
print(f"Min brightness: {min_brightness}")

# Calculate bins centers
n = 12.6  # JND step count
bins_centers = [min_brightness]

while bins_centers[-1] < max_brightness:
    current_bin_center = bins_centers[-1]
    delta_b = get_jnd(current_bin_center)
    next_bin_center = current_bin_center + n * delta_b
    bins_centers.append(next_bin_center)

# Remove the last bin if it exceeds max_brightness
if bins_centers[-1] > max_brightness:
    bins_centers.pop()

print(f"Number of bins: {len(bins_centers)}")
print(f"Bins centers: {bins_centers}")

f_histogram = calculate_histogram_2D(brightness_channel, bins_centers)
print(f"Histogram bins: {f_histogram}")
plt.bar(range(len(bins_centers) - 1), f_histogram, width=1.0)
plt.xlabel('Bin Index')
plt.ylabel('Frequency')
plt.title('Histogram of Brightness Values')
plt.show()

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

# Append the last bin
new_brightness_levels.append(current_brightness)
new_brightness_counts.append(current_count)

print(len(new_brightness_levels))
r_histogram = calculate_histogram_1D(new_brightness_levels, bins_centers)
print(f"Histogram bins: {r_histogram}")

plt.bar(range(len(bins_centers) - 1), r_histogram, width=1.0)
plt.xlabel('Bin Index')
plt.ylabel('Frequency')
plt.title('Histogram of Brightness Values')
plt.show()
