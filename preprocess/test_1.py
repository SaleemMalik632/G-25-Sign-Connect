import cv2
import numpy as np
import pickle
import mediapipe as mp

# Load MediaPipe connections for drawing
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Drawing the keypoints and skeleton on a blank canvas
def draw_keypoints_and_skeleton_on_canvas(keypoints, canvas_size=(640, 480)):
    num_pose_keypoints = 6
    num_hand_keypoints = 21
    num_face_keypoints = 37

    pose_kps = keypoints[:num_pose_keypoints * 3].reshape(num_pose_keypoints, 3)
    left_hand_kps = keypoints[num_pose_keypoints * 3:num_pose_keypoints * 3 + num_hand_keypoints * 3].reshape(num_hand_keypoints, 3)
    right_hand_kps = keypoints[num_pose_keypoints * 3 + num_hand_keypoints * 3:num_pose_keypoints * 3 + 2 * num_hand_keypoints * 3].reshape(num_hand_keypoints, 3)
    face_kps = keypoints[-num_face_keypoints * 3:].reshape(num_face_keypoints, 3)

    # Create a blank canvas (black image)
    canvas = np.zeros((canvas_size[1], canvas_size[0], 3), dtype=np.uint8)

    # Draw skeleton lines for pose
    pose_connections = [(0, 1), (1, 2), (2, 3), (4, 5)]  # Example connections (shoulder to wrist, etc.)
    for connection in pose_connections:
        kp1, kp2 = pose_kps[connection[0]], pose_kps[connection[1]]
        x1, y1 = int(kp1[0] * canvas_size[0]), int(kp1[1] * canvas_size[1])
        x2, y2 = int(kp2[0] * canvas_size[0]), int(kp2[1] * canvas_size[1])
        cv2.line(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Draw pose keypoints
    for kp in pose_kps:
        x, y = int(kp[0] * canvas_size[0]), int(kp[1] * canvas_size[1])
        cv2.circle(canvas, (x, y), 5, (0, 255, 0), -1)

    # Draw skeleton for left hand keypoints
    hand_connections = mp_holistic.HAND_CONNECTIONS  # MediaPipe hand skeleton connections
    for connection in hand_connections:
        kp1, kp2 = left_hand_kps[connection[0]], left_hand_kps[connection[1]]
        x1, y1 = int(kp1[0] * canvas_size[0]), int(kp1[1] * canvas_size[1])
        x2, y2 = int(kp2[0] * canvas_size[0]), int(kp2[1] * canvas_size[1])
        cv2.line(canvas, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # Draw left hand keypoints
    for kp in left_hand_kps:
        x, y = int(kp[0] * canvas_size[0]), int(kp[1] * canvas_size[1])
        cv2.circle(canvas, (x, y), 5, (255, 0, 0), -1)

    # Draw skeleton for right hand keypoints
    for connection in hand_connections:
        kp1, kp2 = right_hand_kps[connection[0]], right_hand_kps[connection[1]]
        x1, y1 = int(kp1[0] * canvas_size[0]), int(kp1[1] * canvas_size[1])
        x2, y2 = int(kp2[0] * canvas_size[0]), int(kp2[1] * canvas_size[1])
        cv2.line(canvas, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # Draw right hand keypoints
    for kp in right_hand_kps:
        x, y = int(kp[0] * canvas_size[0]), int(kp[1] * canvas_size[1])
        cv2.circle(canvas, (x, y), 5, (0, 0, 255), -1)

    # Draw face keypoints
    for i in range(len(face_kps) - 1):
        kp1, kp2 = face_kps[i], face_kps[i + 1]
        x1, y1 = int(kp1[0] * canvas_size[0]), int(kp1[1] * canvas_size[1])
        x2, y2 = int(kp2[0] * canvas_size[0]), int(kp2[1] * canvas_size[1])
        cv2.line(canvas, (x1, y1), (x2, y2), (0, 255, 255), 1)

    # Draw face keypoints
    for kp in face_kps:
        x, y = int(kp[0] * canvas_size[0]), int(kp[1] * canvas_size[1])
        cv2.circle(canvas, (x, y), 3, (0, 255, 255), -1)

    return canvas

def visualize_keypoints(keypoints_path):
    # Load keypoints from pickle file
    with open(keypoints_path, 'rb') as f:
        keypoints = pickle.load(f)

    frame_idx = 0
    while frame_idx < len(keypoints):
        # Create canvas and draw keypoints and skeleton for this frame
        canvas = draw_keypoints_and_skeleton_on_canvas(keypoints[frame_idx])

        # Display the canvas with keypoints and skeleton
        cv2.imshow('Keypoints & Skeleton Visualization', canvas)
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break

        frame_idx += 1

    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Replace this path with the path to your keypoints pickle file
    keypoints_path = r"E:\FYp R And D\implemetation - Copy\preprocess\pose\94pSo75fZrM\OP\0005.pkl"
    
    visualize_keypoints(keypoints_path)
