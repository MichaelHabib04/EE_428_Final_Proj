import cv2 
import numpy as np
import pandas as pd

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





# ------------ Configurables and Constants
cali_sqare_1 =[(810,535),(835,545), 17,(30,100), (160, 255)] #[start postion, end postion,hue threshold,  saturation bound, value bound]
cali_sqare_2 =[(835,535),(855,545), 15,(120,255), (160, 255)] 
first_checkerboard_frame = 240 # since first frame where checkerboard is found is known, skip through video to save 
board_size = (5, 5)   # 6x6 squares -> 5x5 inner corners
outermost_board_corners = None
board_found = False
video_path = 'videos/test3_5_5.h264'


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
origin = point_1
print(point_1)
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


cap = cv2.VideoCapture(video_path) # reinitialize video capture so tracking starts at beginning
first_frame = 0

if not cap.isOpened():
    print("Error opening video file")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break 

    if first_frame == 0:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_pink, upper_pink, color_bgr_pink = color_cali(cali_sqare_1, hsv)
        lower_red, upper_red, color_bgr_red = color_cali(cali_sqare_2, hsv)
        first_frame = 1
    draw_frame = frame.copy() 
    draw_frame, pink_sled_location = draw_rectan_tracker(frame, draw_frame, lower_pink, upper_pink, color_bgr_pink)
    draw_frame, red_sled_location= draw_rectan_tracker(frame, draw_frame, lower_red, upper_red, color_bgr_red)

    cv2.imshow('frame', draw_frame)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()