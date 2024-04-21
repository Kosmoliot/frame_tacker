import cv2
import pytesseract

# Path to Tesseract executable (change accordingly)
pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/Cellar/tesseract/5.3.4_1/bin/tesseract'

# Open the video file or capture video from camera
cap = cv2.VideoCapture('video/sample.mp4')

# Define the coordinates of the region of interest (ROI)
roi_x, roi_y, roi_width, roi_height = 350, 150, 1500, 150

output_file = open('output_file.txt', 'w') 

# Initialize frame counter
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Crop the frame to extract the ROI
    roi = frame[roi_y:roi_y+roi_height, roi_x:roi_x+roi_width]
    
    # Preprocess the frame (convert to grayscale)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # Use Tesseract OCR to recognize numbers
    numbers = pytesseract.image_to_string(gray, config='--psm 6 digits')
    
    output = {}
    # Display or save the recognized numbers
    if numbers:
        output_file.write(f"Frame {frame_count}: Recognized numbers in ROI: {numbers.strip()}\n")
    
    # Increment frame counter
    frame_count += 1
    
    # Display the frame with ROI
    cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_width, roi_y + roi_height), (0, 255, 0), 2)
    cv2.imshow('Video with ROI', frame)
    
    # Press 'q' to exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Close the output file
output_file.close()
# Release resources
cap.release()
cv2.destroyAllWindows()
