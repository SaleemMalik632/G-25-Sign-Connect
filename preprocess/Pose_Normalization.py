import os
import numpy as np
import pickle
import argparse
from glob import glob
from tqdm.auto import tqdm

# Function to normalize 3D skeleton landmarks
def normalize_3D_skeleton_landmarks(keypoints):
    LEFT_SHOULDER_IDX = 0  # Index of left shoulder in keypoints
    RIGHT_SHOULDER_IDX = 1  # Index of right shoulder in keypoints

    normalized_keypoints = []

    for frame in keypoints:
        frame_kps = frame.reshape(-1, 3)

        left_shoulder = frame_kps[LEFT_SHOULDER_IDX]
        right_shoulder = frame_kps[RIGHT_SHOULDER_IDX]

        # Midpoint between shoulders
        shoulder_midpoint = (left_shoulder + right_shoulder) / 2.0

        # Calculate shoulder distance (Euclidean distance)
        shoulder_distance = np.linalg.norm(left_shoulder - right_shoulder)

        # Normalize all keypoints by subtracting the midpoint and scaling by shoulder distance
        normalized_frame = (frame_kps - shoulder_midpoint) / shoulder_distance
        normalized_keypoints.append(normalized_frame.flatten())

    normalized_keypoints = np.array(normalized_keypoints)
    return normalized_keypoints

# Function to load keypoints from a pickle file
def load_keypoints(keypoints_path):
    with open(keypoints_path, 'rb') as f:
        keypoints = pickle.load(f)
    return keypoints

def gen_keypoints_for_video(Pose_path, save_path):
    print('Processing', Pose_path)
    if not os.path.isfile(Pose_path):
        print("SKIPPING MISSING FILE:", Pose_path)
        return
    # Load keypoints from the pickle file
    keypoints = load_keypoints(Pose_path)
    # Normalize the loaded keypoints
    kps_normalization = normalize_3D_skeleton_landmarks(keypoints)
    # Save the normalized keypoints as a pickle file
    with open(save_path, 'wb') as f:
        pickle.dump(kps_normalization, f)

# Main function to handle command-line arguments and run normalization
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-src', '--src_dir', help='Directory containing keypoints files', required=True)
    parser.add_argument('-dest', '--dest_dir', help='Destination directory for normalized keypoints', required=True)
    args = parser.parse_args()

    Poses = [name for name in os.listdir(args.src_dir) if os.path.isdir(os.path.join(args.src_dir, name))]
    for Pose in Poses:
        Pose_name = os.path.splitext(os.path.basename(Pose))[0]
        print(Pose_name)
        Pose_save_dir = os.path.join(args.dest_dir, Pose_name, "OP")
        os.makedirs(Pose_save_dir, exist_ok=True)
        Pose_src_dir = os.path.join(args.src_dir, Pose_name, 'OP')
        print(Pose_src_dir)
        segments = glob(os.path.join(Pose_src_dir, "*.pkl"))
        print(segments)
        for segment in tqdm(segments):
            segment_name = os.path.basename(segment)  # Extracts the filename from the path
            segment_name_without_extension = os.path.splitext(segment_name)[0]  # Removes the extension
            segment_path = os.path.join(Pose_save_dir, segment_name_without_extension + "_normalized.pkl")
            gen_keypoints_for_video(segment, segment_path)
