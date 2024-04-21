import cv2
import pytesseract

# Path to Tesseract executable (change accordingly)
pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/Cellar/tesseract/5.3.4_1/bin/tesseract'

# Open the video file or capture video from camera
cap = cv2.VideoCapture('video/short_sample.mp4')

# Initialize frame counter
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Preprocess the frame (convert to grayscale)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Use Tesseract OCR to recognize numbers
    numbers = pytesseract.image_to_string(gray, config='--psm 6 digits')
    
    output = {}
    # Display or save the recognized numbers
    if numbers:
        output = {frame_count: {numbers.strip()}}
    
    print(output)
    # Increment frame counter
    frame_count += 1
    
    # Display the frame
    cv2.imshow('Video', frame)
    
    # Press 'q' to exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
