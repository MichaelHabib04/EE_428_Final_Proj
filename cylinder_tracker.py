import cv2 
import numpy as np

# ------------ Helper Functions -------
def color_cali(cali_sqare, hsv):
    x1, y1 = cali_sqare[0]
    x2, y2 = cali_sqare[1]

    roi_hue = hsv[y1:y2, x1:x2, 0]
    hue_mean = int(np.round(roi_hue.mean()))
    hue_mean = max(0, min(179, hue_mean))
    hsv_pixel = np.uint8([[[hue_mean, 255, 255]]])
    color_bgr = tuple(int(v) for v in cv2.cvtColor(hsv_pixel, cv2.COLOR_HSV2BGR)[0, 0])

    lower = np.array([hue_mean-threshold,  30, 160], dtype=np.uint8)
    upper = np.array([hue_mean+threshold, 100,   255], dtype=np.uint8)

    return (lower, upper, color_bgr)


def draw_rectan_tracker(frame, lower, upper):
    blur_frame = cv2.blur(frame, (9, 9), 0)
    new_hsv = cv2.cvtColor(blur_frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(new_hsv, lower, upper) 

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        area = cv2.contourArea(c)
        if area < 40: 
            continue
        x, y, w, h = cv2.boundingRect(c)
        if w > 20 :
            if w/h > 2:
                continue
            
        cv2.rectangle(frame, (x,y),  (x+w,y+h), color_bgr_pink, 2)
    return frame




# ------------ Main  -------

cap = cv2.VideoCapture('videos/test3_5_2.h264')

if not cap.isOpened():
    print("Error opening video file")
    exit()


cali_sqare_1 =[(810,535),(835,548)] #[start, end] postion
cali_sqare_2 =[(835,535),(860,548)]
first_frame = 0
threshold = 20
while True:
    ret, frame = cap.read()
    if not ret:
        break 

    if first_frame == 0:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower_pink, upper_pink, color_bgr_pink = color_cali(cali_sqare_1, hsv)
        lower_red, upper_red, color_bgr_red = color_cali(cali_sqare_2, hsv)
        first_frame = 1

    frame = draw_rectan_tracker(frame, lower_pink, upper_pink)

    cv2.imshow('FRAME', frame)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
