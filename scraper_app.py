import cv2
import pytesseract

# Path to Tesseract executable (change accordingly)
pytesseract.pytesseract.tesseract_cmd = '/usr/local/bin/tesseract'

# Open the video file or capture video from camera
cap = cv2.VideoCapture('input_video.mp4')

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
    
    # Display or save the recognized numbers
    if numbers:
        print(f"Frame {frame_count}: Recognized numbers: {numbers.strip()}")
    
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
