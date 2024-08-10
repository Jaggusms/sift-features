import cv2
import numpy as np

# Load the template image
template = cv2.imread('kajal_pic3.png')
if template is None:
    raise ValueError("Check the template image path.")

# Initialize the SIFT detector
sift = cv2.SIFT_create()

# Find keypoints and descriptors in the template image
keypoints_template, descriptors_template = sift.detectAndCompute(template, None)

# Initialize the video capture
cap = cv2.VideoCapture('kajal - Trim1.mp4')

while True:
    # Read a frame from the video
    ret, frame = cap.read()
    if not ret:
        break

    # Find keypoints and descriptors in the frame
    keypoints_frame, descriptors_frame = sift.detectAndCompute(frame, None)

    # Initialize the BFMatcher (Brute Force Matcher)
    bf = cv2.BFMatcher()

    # Match descriptors between the template and frame
    matches = bf.knnMatch(descriptors_template, descriptors_frame, k=2)

    # Apply ratio test to find good matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.85 * n.distance:
            good_matches.append(m)

    # If enough good matches are found, draw a rectangle around the object
    if len(good_matches) > 20:
        # Extract location of good matches in the frame
        points_frame = np.float32([keypoints_frame[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # Extract location of keypoints in the template image
        points_template = np.float32([keypoints_template[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # Find homography
        M, mask = cv2.findHomography(points_template, points_frame, cv2.RANSAC, 5.0)

        # Get the dimensions of the template image
        h, w = template.shape[:2]

        # Define the corners of the template image
        corners = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)

        # Transform the corners of the template image to the frame
        transformed_corners = cv2.perspectiveTransform(corners, M)

        # Draw a rectangle around the template in the frame
        frame_with_rectangle = cv2.polylines(frame, [np.int32(transformed_corners)], True, (0, 255, 0), 2, cv2.LINE_AA)

        # Display the frame with the rectangle
        cv2.imshow('Frame', frame_with_rectangle)

    else:
        cv2.imshow('Frame', frame)

    # Wait for key press to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()
