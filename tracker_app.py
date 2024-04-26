import cv2
import numpy as np
import csv

# Capture video from camera or file
cap = cv2.VideoCapture('video/ski.mp4')

# Parameters for Lucas-Kanade optical flow
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Color for optical flow visualization
color = (0, 255, 0)

# Initialize previous frame
ret, prev_frame = cap.read()
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

# Initialize variables for speed calculation
prev_x = prev_pts[:, 0, 0].mean()


csv_file_path = 'speed_tracker.csv'
with open(csv_file_path, 'w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)

    csv_writer.writerow(['Frame', 'Speed along x-axis'])

    frame_count = 1

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate optical flow
        new_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, frame_gray, prev_pts, None, **lk_params)
        
        # Filter valid points and calculate mean movement on x-axis
        new_pts_valid = new_pts[status == 1]
        new_x = new_pts_valid[:, 0].mean()
        
        # Calculate speed on x-axis
        speed_x = abs(new_x - prev_x)
        
        # Update previous frame and points
        prev_gray = frame_gray.copy()
        prev_pts = new_pts_valid.reshape(-1, 1, 2)
        prev_x = new_x
        
        # Output speed
        csv_writer.writerow([frame_count, speed_x])
        # print("Speed along x-axis:", speed_x)
        
        # Display optical flow
        for p in new_pts_valid:
            x, y = p.ravel()
        cv2.circle(frame, (int(x), int(y)), 5, color, -1)

        cv2.imshow('Optical Flow', frame)

        frame_count += 1
        
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
