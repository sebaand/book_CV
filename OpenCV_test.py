import cv2
import mediapipe as mp
import pytesseract

# Initialize Tesseract-OCR
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Update this path

# Create a VideoCapture object
cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture("http://192.168.1.31:8080/video")
# cap = cv2.VideoCapture("http://[2603:8080:6cf0:72a0:2c9e:1eff:fe90:ecba]:8080/video")

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

# Initialize MediaPipe Drawing utility
mp_drawing = mp.solutions.drawing_utils

while True:
    # Read the current frame from the video stream
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame's color space from BGR to RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Hands
    results = hands.process(frame)

    # Convert the frame's color space back to BGR
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Draw the hand annotations on the frame and check for the "pointingq" gesture
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get the landmarks for the thumb, index finger, middle finger, ring finger, and pinky
            thumb = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            middle = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            ring = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
            pinky = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

            # Check if the index finger is extended and the other fingers are not
            if (index.y < thumb.y and index.y < middle.y and index.y < ring.y and index.y < pinky.y):
                print("Pointing gesture detected")
                # Specify ROI dimensions and extract the ROI from the frame
                roi_size = 100  # Adjust as necessary
                x = int(index.x * frame.shape[1]) - roi_size // 2
                y = int(index.y * frame.shape[0]) - roi_size // 2
                roi = frame[y:y + roi_size, x:x + roi_size]

                # Run OCR on the ROI
                text = pytesseract.image_to_string(roi)
                print("Detected text: ", text)

            else:
                print("not detected")

    # Display the frame
    cv2.imshow('Frame', frame)

    # If the 'q' key is pressed, break from the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the VideoCapture object and close display windows
cap.release()
cv2.destroyAllWindows()
