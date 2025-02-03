import cv2
import mediapipe as mp
import time  # Import time for FPS calculation


class handDetector():
    def __init__(self, mode=False, maxHands=2, model_complexity=1, detectioncon=0.5, trackcon=0.5):
        # Initialize hand tracking parameters
        self.mode = mode
        self.maxHands = maxHands
        self.model_complexity = model_complexity
        self.detectioncon = detectioncon
        self.trackcon = trackcon

        # Initialize Mediapipe Hands module
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands,
                                        self.model_complexity, self.detectioncon, self.trackcon)
        self.mpDraw = mp.solutions.drawing_utils  # Drawing utility for hand landmarks

    def findHand(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert the image to RGB
        self.results = self.hands.process(imgRGB)  # Process the image to detect hands

        if self.results.multi_hand_landmarks:  # Check if any hand landmarks are detected
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)  # Draw hand landmarks
        return img  # Return the processed image

    def findPosition(self, img, handNo=0, draw=True):
        lmList = []  # List to store landmark positions
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]  # Get the specified hand
            for id, lm in enumerate(myHand.landmark):  # Iterate through landmarks
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)  # Convert landmark coordinates to pixel values
                lmList.append([id, cx, cy])  # Store landmark ID and position
                if draw:
                    #if id == 8:  # Check if the landmark is the index fingertip
                    cv2.circle(img, (cx, cy), 7, (255, 0, 0), cv2.FILLED)  # Draw a circle at the fingertip
        return lmList  # Return the list of landmark positions


def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)  # Start video capture from webcam
    detector = handDetector()  # Create an instance of handDetector
    while True:
        success, img = cap.read()  # Read a frame from the webcam
        img = detector.findHand(img)  # Detect and draw hands
        lmList = detector.findPosition(img)  # Get landmark positions
        # if len(lmList) != 0:
        #     print(lmList[8])  # Print the coordinates of the index fingertip

        # Calculate Frames Per Second (FPS)
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        # Display FPS on the screen
        cv2.putText(img, str(int(fps)), (10, 78), cv2.FONT_HERSHEY_PLAIN,
                    3, (255, 0, 255), 3)

        cv2.imshow("Image", img)  # Show the processed image
        cv2.waitKey(1)  # Wait for a key press (1ms delay for real-time processing)


if __name__ == "__main__":
    main()  # Run the main function when the script is executed
