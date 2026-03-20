import cv2 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------ Helper Functions -------
def color_cali(cali_sqare, hsv):
    x1, y1 = cali_sqare[0]
    x2, y2 = cali_sqare[1]
    threshold = cali_sqare[2]
    sat_lower, sat_upper = cali_sqare[3]
    val_lower, val_upper = cali_sqare[4]

    roi_hue = hsv[y1:y2, x1:x2, 0]
    hue_mean = int(np.round(roi_hue.mean()))
    hue_mean = max(0, min(179, hue_mean))
    hsv_pixel = np.uint8([[[hue_mean, 255, 255]]])
    color_bgr = tuple(int(v) for v in cv2.cvtColor(hsv_pixel, cv2.COLOR_HSV2BGR)[0, 0])
    lower = np.array([hue_mean-threshold,  sat_lower, val_lower], dtype=np.uint8)
    upper = np.array([hue_mean+threshold, sat_upper,   val_upper], dtype=np.uint8)
    return (lower, upper, color_bgr)

def draw_rectan_tracker(frame, draw_frame, lower, upper, color):
    blur_frame = cv2.blur(frame, (9, 9), 0)
    new_hsv = cv2.cvtColor(blur_frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(new_hsv, lower, upper) 

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    location = None
    for c in contours:
        area = cv2.contourArea(c) # check area
        if area < 30: 
            continue
        if area > 140: # reject calibration mats
            continue
        x, y, w, h = cv2.boundingRect(c)
        if w > 10 : # check aspect ratio
            if w/h > 3:
                continue
        location = (x + w/2, y + h/2) # record center of bounding box as sled location
        cv2.rectangle(draw_frame, (x,y),  (x+w,y+h), color, 2)
        break # only one sled exists in each frame
    return draw_frame, location

def pixel_to_Cam_Space(
    uv_undistorted,
    H,
    origin_uv_source,
    axis_p1_source,
    axis_p2_source,
    mat_width_in=24,
    mat_height_in=24,
):
    """
    Map a pixel position in the homography-warped image to X/Y in inches from origin marker.

    +X is defined by axis_p1_source -> axis_p2_source (in SOURCE image), after homography.
    """
    H = np.asarray(H, dtype=np.float64)

    origin_w = apply_homography_to_point(H, origin_uv_source)
    p1_w = apply_homography_to_point(H, axis_p1_source)
    p2_w = apply_homography_to_point(H, axis_p2_source)

    v_x = (p2_w - p1_w)
    len_x_px = float(np.linalg.norm(v_x))
    if len_x_px < 1e-9:
        raise ValueError("Axis points are too close after homography. Cannot define x axis.")
    ex = v_x / len_x_px

    # Perpendicular in image coords (+u right, +v down)
    ey = np.array([ex[1], -ex[0]], dtype=np.float64)

    p_w = np.array([float(uv_undistorted[0]), float(uv_undistorted[1])], dtype=np.float64)
    d = p_w - origin_w
    dx_px = float(d.dot(ex))
    dy_px = float(d.dot(ey))

    # Scale (inches per pixel)
    m_per_px_x = float(mat_width_in) / len_x_px
    len_y_px = len_x_px * (float(mat_height_in) / float(mat_width_in))
    m_per_px_y = float(mat_height_in) / len_y_px

    x_m = dx_px * m_per_px_x
    y_m = dy_px * m_per_px_y
    return float(x_m), float(y_m)

def apply_homography_to_point(H, uv):
    u, v = float(uv[0]), float(uv[1])
    p = np.array([u, v, 1.0], dtype=np.float64)
    q = H @ p
    if abs(q[2]) < 1e-12:
        raise ValueError("Homography produced point at infinity (w ~ 0).")
    return np.array([q[0] / q[2], q[1] / q[2]], dtype=np.float64)

def pixel_source_to_world_xy_in(
    uv_source,
    H,
    origin_uv_source,
    axis_p1_source,
    axis_p2_source,
    mat_width_in=24,
    mat_height_in=24,
):
    """
    (source pixel) -> (warped pixel) -> (x,y) meters from origin
    """
    uv_warped = apply_homography_to_point(H, uv_source)
    return pixel_to_Cam_Space(
        uv_undistorted=uv_warped,
        H=H,
        origin_uv_source=origin_uv_source,
        axis_p1_source=axis_p1_source,
        axis_p2_source=axis_p2_source
    )

# ------------ Configurables and Constants
cali_sqare_1 =[(810,535),(835,545), 17,(30,100), (160, 255)] #[start postion, end postion,hue threshold,  saturation bound, value bound]
cali_sqare_2 =[(835,535),(855,545), 15,(120,255), (160, 255)] 
first_checkerboard_frame = 240 # since first frame where checkerboard is found is known, skip through video to save 
board_size = (5, 5)   # 6x6 squares -> 5x5 inner corners
outermost_board_corners = None
board_found = False
video_path = 'videos/test3_5_5.h264'
FPS = 30 # camera framerate


# ------------ Main  -------
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error opening video file")
    exit()

print("Finding Checkerboard")
for i in range(first_checkerboard_frame):
    cap.read()

while not board_found:
    ret, frame = cap.read()
    if not ret:
        print("Failed to locate board in video")
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCornersSB(gray, board_size) # identify chessboard corner locations with subpixel accuracy
    if found:
        corners = corners.reshape(-1, 2)
        # Store outermost corners
        top_left = corners[0]
        top_right = corners[4]
        bottom_left = corners[-5]
        bottom_right = corners[-1]
        outermost_board_corners = [top_left, top_right, bottom_left, bottom_right]
        board_found = True
        cv2.drawChessboardCorners(frame, board_size, corners, found)
        for corner in outermost_board_corners:
            point = (int(corner[0]), int(corner[1]))
            cv2.circle(frame, center=point, radius=20, color=(0, 255, 0), thickness=2)
            # Draw a red star marker on each outer corner
            cv2.drawMarker(frame, point, (0, 0, 255), 
               markerType=cv2.MARKER_STAR, 
               markerSize=10, 
               thickness=2)
        board_finder_preview = cv2.namedWindow("Board Finder")
        cv2.imshow("Board Finder",frame)
        cv2.waitKey(0)

# calculate destination points for homography (normal view of the marker)
point_1 = np.array(bottom_left)
point_2 = np.array(top_left)
point_3 = np.array(top_right)
point_4 = np.array(bottom_right)
# select origin to measure from
origin_pix = bottom_left
print("pixel origin: ", origin_pix)
mat_aspect_ratio = 1
# calculate destination points for homography (normal view of the mat)
v1 = point_2 - point_1
v1 = np.append(v1, 0)  # make it 3D by adding a z component
v2 = np.cross(v1, [0, 0, -1])  # perpendicular vector in the plane
v2 = v2 / mat_aspect_ratio
v2 = np.array(v2[:2], dtype=np.float32)  # convert back to 2D and ensure float32 type
source_points = np.array([point_1, point_2, point_3, point_4], dtype=np.float32).reshape(-1, 1, 2)
dest_points = np.array([point_1, point_2, point_2 + v2, point_1 + v2], dtype=np.float32).reshape(-1, 1, 2)
# destination points for homography (normal view of the marker), should be calculate to match shape of the true mat
homography, _ = cv2.findHomography(source_points, dest_points, method=cv2.RANSAC)
undistorted = cv2.warpPerspective(frame, homography, (1300,700))
cv2.polylines(frame, np.array([source_points], dtype=np.int32), True, (0, 255, 255), 3)
cv2.imshow("Checkerboard edges", frame)
cv2.polylines(undistorted, np.array([dest_points], dtype=np.int32), True, (0, 255, 255), 3)
cv2.imshow("Projected Top-Down View", undistorted)
cv2.waitKey(0)


origin_real_space = pixel_source_to_world_xy_in(origin_pix, homography, origin_pix, bottom_left, bottom_right, homography)
print("REAL ORIGIN: ", origin_real_space)

cap = cv2.VideoCapture(video_path) # reinitialize video capture so tracking starts at beginning
first_frame = 0
record = pd.DataFrame(columns=['Frame number', 
                                    'Time', 
                                    'Pink Sled Pixel Location', 
                                    'Red Sled Pixel Location',
                                    'Pink Sled Real Location', 
                                    'Red Sled Real Location'])


if not cap.isOpened():
    print("Error opening video file")
    exit()

frame_count = 0
frame_window = cv2.namedWindow('frame')
undist_window = cv2.namedWindow('undistorted frame')

last_pink_sled_location = origin_pix # Initially, if sleds are not found, return origin pixel values
last_red_sled_location = origin_pix
while True:
    ret, frame = cap.read()
    if not ret:
        break 
    
    if first_frame == 0:
        # run color calibration on first frame
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_pink, upper_pink, color_bgr_pink = color_cali(cali_sqare_1, hsv)
        lower_red, upper_red, color_bgr_red = color_cali(cali_sqare_2, hsv)
        first_frame = 1

    draw_frame = frame.copy() 
    # draw bounding boxes around sleds
    draw_frame, pink_sled_location = draw_rectan_tracker(frame, draw_frame, lower_pink, upper_pink, color_bgr_pink)
    draw_frame, red_sled_location = draw_rectan_tracker(frame, draw_frame, lower_red, upper_red, color_bgr_red)
    # if sled was not found, record last known location
    if pink_sled_location:
        last_pink_sled_location = pink_sled_location
    else:
        pink_sled_location = last_pink_sled_location

    if red_sled_location:
        last_red_sled_location = red_sled_location
    else:
        red_sled_location = last_red_sled_location

    pink_loc_pix_x = pink_sled_location[0]
    pink_loc_pix_y = pink_sled_location[1]
    red_loc_pix_x = red_sled_location[0]
    red_loc_pix_y = red_sled_location[1]

    # Determine the real space in pixels
    pink_loc_real = pixel_source_to_world_xy_in((pink_loc_pix_x, pink_loc_pix_y), homography, origin_pix, bottom_left, bottom_right)
    red_loc_real = pixel_source_to_world_xy_in((red_loc_pix_x, red_loc_pix_y), homography, origin_pix ,bottom_left, bottom_right)
    new_row = pd.DataFrame({"Frame number": [frame_count], 
                            "Time": [frame_count/30], 
                            "Pink Sled Pixel Location": [pink_sled_location], 
                            "Red Sled Pixel Location": [red_sled_location], 
                            "Pink Sled Real Location": [pink_loc_real], 
                            "Red Sled Real Location": [red_loc_real]})
    # Concatenate the original DataFrame and the new row DataFrame
    record = pd.concat([record, new_row], ignore_index=True)

    frame_w_origin = frame.copy()
    cv2.drawMarker(frame_w_origin, 
                (int(origin_pix[0]), int(origin_pix[1])), 
                (0, 0, 255), 
               markerType=cv2.MARKER_STAR, 
               markerSize=10, 
               thickness=2)
    draw_undist_frame = cv2.warpPerspective(frame_w_origin, homography, (1300,700))
    cv2.putText(draw_undist_frame, f"pink location={pink_loc_real}", (30, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(draw_undist_frame, f"red location={red_loc_real}", (30, 200),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('frame', draw_frame)
    cv2.imshow('undistorted frame', draw_undist_frame)
    frame_count += 1
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# plot real positions from dataframe
if not record.empty:
    plot_time = pd.to_numeric(record['Time'], errors='coerce')

    def _split_xy(series):
        x_vals = []
        y_vals = []
        for value in series:
            if isinstance(value, (tuple, list, np.ndarray)) and len(value) >= 2:
                x_vals.append(float(value[0]))
                y_vals.append(float(value[1]))
            else:
                x_vals.append(np.nan)
                y_vals.append(np.nan)
        return np.array(x_vals, dtype=float), np.array(y_vals, dtype=float)

    pink_x, pink_y = _split_xy(record['Pink Sled Real Location'])
    red_x, red_y = _split_xy(record['Red Sled Real Location'])

    plt.figure()
    plt.plot(plot_time, pink_x)
    plt.xlabel('Time (s)')
    plt.ylabel('Pink sled X position (inches)')
    plt.title('Pink sled X position vs time')
    plt.grid(True)
    plt.tight_layout()

    plt.figure()
    plt.plot(plot_time, pink_y)
    plt.xlabel('Time (s)')
    plt.ylabel('Pink sled Y position (inches)')
    plt.title('Pink sled Y position vs time')
    plt.grid(True)
    plt.tight_layout()

    plt.figure()
    plt.plot(plot_time, red_x)
    plt.xlabel('Time (s)')
    plt.ylabel('Red sled X position (inches)')
    plt.title('Red sled X position vs time')
    plt.grid(True)
    plt.tight_layout()

    plt.figure()
    plt.plot(plot_time, red_y)
    plt.xlabel('Time (s)')
    plt.ylabel('Red sled Y position (inches)')
    plt.title('Red sled Y position vs time')
    plt.grid(True)
    plt.tight_layout()

    plt.show()