import cv2 
import numpy as np

cap = cv2.VideoCapture('videos/test3_5_2.h264')


if not cap.isOpened():
    print("Error opening video file")
    exit()


cali_sqare_1 =[(810,535),(835,548)] #[start, end] postion
cali_sqare_2 =[(835,535),(860,548)]
first_frame = 0
threshold = 20
while True:
    ret, new_frame = cap.read()
    frame = new_frame
    if not ret:
        break 

    if first_frame == 0:
        untocie = frame
        h, w = frame.shape[:2]
        frame = cv2.blur(frame, (9, 9), 0)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        x1, y1 = cali_sqare_1[0]
        x2, y2 = cali_sqare_1[1]

        roi_hue = hsv[y1:y2, x1:x2, 0]
        hue_mean = int(np.round(roi_hue.mean()))
        hue_mean = max(0, min(179, hue_mean))
        hsv_pixel = np.uint8([[[hue_mean, 255, 255]]])
        color_bgr_1 = tuple(int(v) for v in cv2.cvtColor(hsv_pixel, cv2.COLOR_HSV2BGR)[0, 0])

        lower_pink = np.array([hue_mean-threshold,  30, 160], dtype=np.uint8)
        upper_pink = np.array([hue_mean+threshold, 100,   255], dtype=np.uint8)

        x1, y1 = cali_sqare_2[0]
        x2, y2 = cali_sqare_2[1]

        roi_hue = hsv[y1:y2, x1:x2, 0]
        hue_mean = int(np.round(roi_hue.mean()))
        hue_mean = max(0, min(179, hue_mean))
        hsv_pixel = np.uint8([[[hue_mean, 255, 255]]])
        color_bgr_1 = tuple(int(v) for v in cv2.cvtColor(hsv_pixel, cv2.COLOR_HSV2BGR)[0, 0])

        lower_red = np.array([hue_mean-threshold,  30, 160], dtype=np.uint8)
        upper_red = np.array([hue_mean+threshold, 100,   255], dtype=np.uint8)

        first_frame = 1

    frame = cv2.blur(new_frame, (9, 9), 0)
    new_hsv = cv2.cvtColor(new_frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(new_hsv, lower_pink, upper_pink) 

    result = cv2.bitwise_and(new_frame, new_frame, mask=mask)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        area = cv2.contourArea(c)
        if area < 40: 
            continue
        x, y, w, h = cv2.boundingRect(c)
        if w > 20 :
            if w/h > 2:
                continue
            
        cv2.rectangle(new_frame, (x,y),  (x+w,y+h), color_bgr_1, 2)

    # mask = cv2.inRange(new_hsv, lower_red, upper_red) 

    # result = cv2.bitwise_and(new_frame, new_frame, mask=mask)
    # contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # for c in contours:
    #     area = cv2.contourArea(c)
    #     if area < 40: 
    #         continue
    #     x, y, w, h = cv2.boundingRect(c)
    #     if w > 20 :
    #         if w/h > 2:
    #             continue
            
    #     cv2.rectangle(new_frame, (x,y),  (x+w,y+h), color_bgr_1, 2)

    cv2.imshow('FRAME', new_frame)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
