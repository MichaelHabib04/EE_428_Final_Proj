import cv2
import numpy as np
from sklearn.mixture import GaussianMixture


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

