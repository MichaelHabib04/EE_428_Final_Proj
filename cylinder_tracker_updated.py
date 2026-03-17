import cv2
import numpy as np
from sklearn.mixture import GaussianMixture

# ------------ Helper Functions -------
def first_frame_detection(frame):
    np.random.seed(32905)

    num_sleds = 2
    board_size = (5, 5)   # 6x6 squares -> 5x5 inner corners
    y_buffer = 0.10
    x_buffer = 0.05
    num_clusters = num_sleds + 4

    frame_bgr = frame

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    height, width, _ = frame_rgb.shape

    # Detect checkerboard
    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    found, corners = cv2.findChessboardCornersSB(gray, board_size)
    if not found:
        raise RuntimeError("Checkerboard not found in first frame.")
    corners = corners.reshape(-1, 2)

    # Crop region around checkerboard
    top_left = corners[0]
    bottom_right = corners[-1]

    x1 = max(0, int(np.floor(top_left[0] - width * y_buffer)))
    y1 = max(0, int(np.floor(top_left[1] - height * x_buffer)))
    x2 = min(width, int(np.floor(bottom_right[0] + width * y_buffer)))
    y2 = min(height, int(np.floor(bottom_right[1] + height * x_buffer)))

    color_patches = frame_rgb[y1:y2, x1:x2].copy()
    if color_patches.size == 0:
        raise RuntimeError("Patch crop is empty.")

    # Smooth patch image
    kernel = np.ones((5, 5), dtype=np.float32) / 25.0
    color_patches = cv2.filter2D(color_patches, -1, kernel)

    # Build feature matrix: [H, S, normalized R, normalized G, normalized B]
    patch_float = color_patches.astype(np.float64)
    R = patch_float[:, :, 0]
    G = patch_float[:, :, 1]
    B = patch_float[:, :, 2]

    denom = R + G + B
    denom[denom == 0] = 1e-8
    rgb_norm = patch_float / denom[:, :, None]

    patch_hsv = cv2.cvtColor(color_patches, cv2.COLOR_RGB2HSV).astype(np.float64)
    patch_hsv[:, :, 0] /= 179.0
    patch_hsv[:, :, 1] /= 255.0
    patch_hsv[:, :, 2] /= 255.0

    h = patch_hsv[:, :, 0]
    h[h > 0.5] = 1.0 - h[h > 0.5]
    patch_hsv[:, :, 0] = h

    hs = patch_hsv[:, :, :2].reshape(-1, 2)
    rgb_features = rgb_norm.reshape(-1, 3)
    feature_matrix = np.hstack((hs, rgb_features))

    # Cluster patch colors with GMM
    gmm = GaussianMixture(
        n_components=num_clusters,
        reg_covar=0.001,
        random_state=32905
    )
    gmm.fit(feature_matrix)

    labels = gmm.predict(feature_matrix)
    labeled_patches = labels.reshape(color_patches.shape[:2])
    centroids = gmm.means_

    # Select the most saturated valid clusters
    cluster_ids = np.arange(num_clusters).reshape(-1, 1)
    centroids_with_labels = np.hstack((centroids, cluster_ids))

    saturation = centroids_with_labels[:, 1]
    valid = centroids_with_labels[(saturation > 0.05) & (saturation < 0.95)]
    valid = valid[np.argsort(-valid[:, 1])]

    if len(valid) < num_sleds:
        raise RuntimeError("Not enough valid saturated clusters found.")

    # Compute RGB and HSV means for the selected clusters
    patch_hsv_u8 = cv2.cvtColor(color_patches, cv2.COLOR_RGB2HSV)
    rgb_means = []
    hsv_means = []

    for i in range(num_sleds):
        label_id = int(valid[i, 5])
        mask = labeled_patches == label_id

        avg_r = int(np.round(np.mean(color_patches[:, :, 0][mask])))
        avg_g = int(np.round(np.mean(color_patches[:, :, 1][mask])))
        avg_b = int(np.round(np.mean(color_patches[:, :, 2][mask])))
        rgb_means.append((avg_r, avg_g, avg_b))

        avg_h = int(np.round(np.mean(patch_hsv_u8[:, :, 0][mask])))
        avg_s = int(np.round(np.mean(patch_hsv_u8[:, :, 1][mask])))
        avg_v = int(np.round(np.mean(patch_hsv_u8[:, :, 2][mask])))
        hsv_means.append((avg_h, avg_s, avg_v))

    # Output only the means
    for i in range(num_sleds):
        print(f"Cluster {i+1} mean RGB: {rgb_means[i]}")
        print(f"Cluster {i+1} mean HSV: {hsv_means[i]}")

    return rgb_means, hsv_means


def remove_hue_overlap(hsv_means, hue_threshold):
    """
    Build hue intervals around each mean, then trim overlaps.
    Interval sizes may shrink.
    """
    intervals = []
    for i, hsv in enumerate(hsv_means):
        h = int(hsv[0])
        lower = max(0, h - hue_threshold)
        upper = min(179, h + hue_threshold)
        intervals.append([i, h, lower, upper])

    # Sort by hue center
    intervals.sort(key=lambda x: x[1])

    # Trim overlaps between neighbors
    for i in range(len(intervals) - 1):
        curr = intervals[i]
        nxt = intervals[i + 1]

        if curr[3] >= nxt[2]:
            split = (curr[1] + nxt[1]) // 2
            curr[3] = min(curr[3], split)
            nxt[2] = max(nxt[2], split + 1)

            # Safety in case trimming inverted a range
            if curr[3] < curr[2]:
                curr[3] = curr[2]
            if nxt[2] > nxt[3]:
                nxt[2] = nxt[3]

    hue_lowers = [0] * len(hsv_means)
    hue_uppers = [0] * len(hsv_means)

    for orig_idx, _, lower, upper in intervals:
        hue_lowers[orig_idx] = lower
        hue_uppers[orig_idx] = upper

    return hue_lowers, hue_uppers


# Returns thresholding values for both rgb and hsv
def color_cali(hsv_means, thresholds):
    hsv_lowers = []
    hsv_uppers = []

    # Remove hue overlap here
    hue_lowers, hue_uppers = remove_hue_overlap(hsv_means, thresholds[0])

    for i, hsv in enumerate(hsv_means):
        hsv_lowers.append(np.array([
            hue_lowers[i],
            max(0, hsv[1] - thresholds[1]),
            max(0, hsv[2] - thresholds[2])
        ], dtype=np.uint8))

        hsv_uppers.append(np.array([
            hue_uppers[i],
            min(255, hsv[1] + thresholds[1]),
            min(255, hsv[2] + thresholds[2])
        ], dtype=np.uint8))

    print(f"hsv lowers: {hsv_lowers}, hsv uppers: {hsv_uppers}")
    return hsv_lowers, hsv_uppers


def draw_rectan_tracker(frame, draw_frame, lower, upper, color):
    blur_frame = cv2.blur(frame, (9, 9), 0)
    new_hsv = cv2.cvtColor(blur_frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(new_hsv, lower, upper)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        area = cv2.contourArea(c)
        if area < 30:
            continue
        x, y, w, h = cv2.boundingRect(c)
        if w > 10:
            if w / h > 3:
                continue
        bgr_color = [color[2], color[1], color[0]]
        cv2.rectangle(draw_frame, (x, y), (x + w, y + h), bgr_color, 2)
    return draw_frame


# ------------ Main  -------

cap = cv2.VideoCapture('videos/test3_5_3.h264')

if not cap.isOpened():
    print("Error opening video file")
    exit()

thresholds = [18, 40, 255]   # [hue, sat, value]
first_frame = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    if first_frame == 0:
        try:
            rgb_means, hsv_means = first_frame_detection(frame)
            hsv_lowers, hsv_uppers = color_cali(hsv_means, thresholds)
            first_frame = 1
        except:
            first_frame = 0
            continue

    draw_frame = frame.copy()

    # HSV Thresholding
    for i, mean in enumerate(rgb_means):
        draw_frame = draw_rectan_tracker(frame, draw_frame, hsv_lowers[i], hsv_uppers[i], mean)

    cv2.imshow('frame', draw_frame)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()