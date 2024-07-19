
import os
import constants
from feat import Detector
import torch

def preprocess_videos(video_folder, output_folder, skip_frames=None):
    """
    Applies the preprocessing to all videos in the folder. Action Units are computed for each frame of each video and then stored in csv files.
    Arguments:
        video_folder (string): path to the folder containing the videos.
        output_folder (string): path to the folder where the .csv files are stored.
    """
    video_paths = os.listdir(video_folder)[::-1]
    video_df_list = []
    print(f'Processing videos in {video_folder}')
    for i, video_file in enumerate(video_paths):
        if os.path.isfile(f'{output_folder}/{video_file[:-4]}.csv'):
            print(video_file)
        if video_file.endswith(".mp4") and not(os.path.isfile(f'{output_folder}/{video_file[:-4]}.csv')):
            # Construct the full path to the video file
            input_video_path = os.path.join(video_folder, video_file)
            detector = Detector()
            video_prediction = detector.detect_video(input_video_path, skip_frames = skip_frames)
            video_prediction = video_prediction.reset_index(drop=True).fillna(0)
            columns_kept = ['FaceRectX', 'FaceRectY', 'FaceRectWidth', 'FaceRectHeight', 'FaceScore'] + [x for x in constants.ACTION_UNITS.keys()] + ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise', 'neutral', 'input', 'frame', 'approx_time']
            video_df_list.append(video_prediction[columns_kept])
            video_prediction[columns_kept].to_csv(f'{output_folder}/{video_file[:-4]}.csv')
        print(f'File {i}/{len(video_paths)} processed')

preprocess_videos('/home/ens/AU58490/work/YP/videos/clips', '/home/ens/AU58490/work/YP/videos/clips/csvs')
#trimmed_video_path = '/datasets/C-EXPR-DB/trimmed_videos'
#for compound_emotion in ['Angrily-Surprised', 'Disgustedly-Surprised', 'Fearfully-Surprised', 'Happily-Surprised', 'Other', 'Sadly-Angry', 'Sadly-Fearful', 'Sadly-Surprised']:
#    preprocess_videos(video_folder=f'{trimmed_video_path}/{compound_emotion}', output_folder=f'/home/ens/AU58490/work/YP/videos/video_features/{compound_emotion}')