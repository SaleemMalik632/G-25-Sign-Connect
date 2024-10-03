
import cv2
import numpy as np
import pickle
import os
import mediapipe as mp

# Load MediaPipe connections for drawing
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Drawing the keypoints and skeleton on the image
def draw_keypoints_and_skeleton_on_frame(frame, keypoints):
    num_pose_keypoints = 6
    num_hand_keypoints = 21
    num_face_keypoints = 37

    pose_kps = keypoints[:num_pose_keypoints * 3].reshape(num_pose_keypoints, 3)
    left_hand_kps = keypoints[num_pose_keypoints * 3:num_pose_keypoints * 3 + num_hand_keypoints * 3].reshape(num_hand_keypoints, 3)
    right_hand_kps = keypoints[num_pose_keypoints * 3 + num_hand_keypoints * 3:num_pose_keypoints * 3 + 2 * num_hand_keypoints * 3].reshape(num_hand_keypoints, 3)
    face_kps = keypoints[-num_face_keypoints * 3:].reshape(num_face_keypoints, 3)

    # Draw skeleton lines for pose
    pose_connections = [(0, 1), (1, 2), (2, 3), (4, 5)]  # shoulder to wrist and hip to knee
    for connection in pose_connections:
        kp1, kp2 = pose_kps[connection[0]], pose_kps[connection[1]]
        x1, y1 = int(kp1[0] * frame.shape[1]), int(kp1[1] * frame.shape[0])
        x2, y2 = int(kp2[0] * frame.shape[1]), int(kp2[1] * frame.shape[0])
        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Draw pose keypoints
    for kp in pose_kps:
        x, y = int(kp[0] * frame.shape[1]), int(kp[1] * frame.shape[0])
        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

    # Draw skeleton for left and right hand keypoints
    hand_connections = mp_holistic.HAND_CONNECTIONS  # pre-defined connections for hand skeleton
    for connection in hand_connections:
        kp1, kp2 = left_hand_kps[connection[0]], left_hand_kps[connection[1]]
        x1, y1 = int(kp1[0] * frame.shape[1]), int(kp1[1] * frame.shape[0])
        x2, y2 = int(kp2[0] * frame.shape[1]), int(kp2[1] * frame.shape[0])
        cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # Draw left hand keypoints
    for kp in left_hand_kps:
        x, y = int(kp[0] * frame.shape[1]), int(kp[1] * frame.shape[0])
        cv2.circle(frame, (x, y), 5, (255, 0, 0), -1)

    for connection in hand_connections:
        kp1, kp2 = right_hand_kps[connection[0]], right_hand_kps[connection[1]]
        x1, y1 = int(kp1[0] * frame.shape[1]), int(kp1[1] * frame.shape[0])
        x2, y2 = int(kp2[0] * frame.shape[1]), int(kp2[1] * frame.shape[0])
        cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # Draw right hand keypoints
    for kp in right_hand_kps:
        x, y = int(kp[0] * frame.shape[1]), int(kp[1] * frame.shape[0])
        cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)

    # Draw face keypoints (skeleton drawing is optional for face)
    for i in range(len(face_kps) - 1):
        kp1, kp2 = face_kps[i], face_kps[i + 1]
        x1, y1 = int(kp1[0] * frame.shape[1]), int(kp1[1] * frame.shape[0])
        x2, y2 = int(kp2[0] * frame.shape[1]), int(kp2[1] * frame.shape[0])
        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 1)

    # Draw face keypoints
    for kp in face_kps:
        x, y = int(kp[0] * frame.shape[1]), int(kp[1] * frame.shape[0])
        cv2.circle(frame, (x, y), 3, (0, 255, 255), -1)

    return frame

def visualize_keypoints_on_video(video_path, keypoints_path):
    # Load keypoints from pickle file
    with open(keypoints_path, 'rb') as f:
        keypoints = pickle.load(f)

    # Load the video frames
    vidcap = cv2.VideoCapture(video_path)
    frame_idx = 0

    while vidcap.isOpened():
        success, frame = vidcap.read()
        if not success:
            break

        # Ensure we have keypoints for this frame
        if frame_idx < len(keypoints):
            keypoint = keypoints[frame_idx]
            frame = draw_keypoints_and_skeleton_on_frame(frame, keypoint)

        # Display the frame with keypoints and skeleton
        cv2.imshow('Keypoints & Skeleton Visualization', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

        frame_idx += 1

    vidcap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = r"E:\FYp R And D\implemetation - Copy\preprocess\destination\94pSo75fZrM\0005.mp4"
    keypoints_path = r"E:\FYp R And D\implemetation - Copy\preprocess\pose\94pSo75fZrM\OP\0005.pkl"

    visualize_keypoints_on_video(video_path, keypoints_path)