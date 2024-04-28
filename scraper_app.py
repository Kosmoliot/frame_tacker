import cv2
import pytesseract
import csv
import re

# Path to Tesseract executable (change accordingly)
pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/Cellar/tesseract/5.3.4_1/bin/tesseract'

# Open the video file or capture video from camera
video_path = 'video/full_screen.mp4'
cap = cv2.VideoCapture(video_path)

# Define the coordinates of the region of interest (ROI)
roi_x, roi_y, roi_width, roi_height = 425, 950, 750, 75

# Open a CSV file for writing the output
csv_file_path = 'number_capture.csv'
with open(csv_file_path, 'w', newline='') as csvfile:
    fieldnames = ['Frame', '0 axis', '1 axis', '2 axis', '3 axis']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    # Initialize frame counter
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Crop the frame to extract the region of interest (ROI)
        roi = frame[roi_y:roi_y + roi_height, roi_x:roi_x + roi_width]
        
        # Preprocess the ROI (convert to grayscale)
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Use Tesseract OCR to recognize numbers
        numbers = pytesseract.image_to_string(gray_roi, config='--psm 6 digits')
        
        # Write the frame number and recognized numbers to the CSV file
        if numbers:
            # Extract numbers using regular expression, handling lack of gaps
            axis_values = re.findall(r'-?\d+\.\d+', numbers.strip())
            if len(axis_values) >= 4:
                writer.writerow({'Frame': frame_count, '0 axis': axis_values[0], '1 axis': axis_values[1], '2 axis': axis_values[2], '3 axis': axis_values[3]})
        
        # Increment frame counter
        frame_count += 1
        
        # Display the frame with ROI
        cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_width, roi_y + roi_height), (0, 255, 0), 2)
        cv2.imshow('Video with ROI', frame)
        
        # Press 'q' to exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

# Release resources
cap.release()
cv2.destroyAllWindows()
