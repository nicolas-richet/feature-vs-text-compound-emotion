from feat import Detector
import os
import constants
import torch.nn as nn
import torch
from transformers import Wav2Vec2Processor, WhisperProcessor, WhisperForConditionalGeneration
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
)
import librosa
import subprocess
import numpy as np
import pandas as pd
from hume import HumeBatchClient, TranscriptionConfig
from hume.models.config import ProsodyConfig
import json



def main(gpu_id, dataset, splits= ['train', 'test', 'dev']):
    if torch.cuda.is_available():
        torch.cuda.set_device(f'cuda:{gpu_id}')
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    dirs = []
    if dataset == 'MELD':
        data_dir = constants.MELD_DATA_DIR
        detect_voice = False
        dirs = [os.path.join(data_dir, x) for x in splits]
    elif dataset == 'C-EXPR-DB':
        detect_voice = True
        dirs = [constants.C_EXPR_DATA_DIR]
    for dir in dirs:
        video_folder=os.path.join(dir, 'videos')
        print('Processing videos...')
        preprocess_videos(video_folder=video_folder, skip_frames=None, output_folder=os.path.join(dir, 'hume_features'))
        print("Processing Audios...")
        preprocess_audios(video_folder=video_folder, output_folder=dir, device=device, detect_voice=detect_voice)
        print("Analyzing Tone from videos...")
        analyze_tone_from_videos(video_folder, output_dir=os.path.join(data_dir, 'audios'))
        extract_hume_features(input_folder=os.path.join(data_dir, 'audios'), output_folder=os.path.join(data_dir, 'hume_features'))

    
    

def preprocess_videos(video_folder, skip_frames, output_folder):
    """
    Applies the preprocessing to all videos in the folder. Action Units are computed for each video, then the max value (over time) for each AU is kept.

    """
    video_paths = os.listdir(video_folder)
    video_df_list = []
    print(f'Processing videos in {video_folder}')
    for i, video_file in enumerate(video_paths):
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


def preprocess_audios(video_folder, output_folder, device, detect_voice = False):
    """
    Computes Valence-Arousal-Dominance audio features for all videos in the folder.
    Args:
        video_folder (string): path of the video folder. 
        device: device used for the audio preprocessing.
        detect_voice (boolean): if True then a voice activity is used and clips are transcribed.
            Default: False
    """
    
    class RegressionHead(nn.Module):
        r"""Classification head."""

        def __init__(self, config):

            super().__init__()

            self.dense = nn.Linear(config.hidden_size, config.hidden_size)
            self.dropout = nn.Dropout(config.final_dropout)
            self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

        def forward(self, features, **kwargs):

            x = features
            x = self.dropout(x)
            x = self.dense(x)
            x = torch.tanh(x)
            x = self.dropout(x)
            x = self.out_proj(x)

            return x

    class EmotionModel(Wav2Vec2PreTrainedModel):
        r"""Speech emotion classifier."""

        def __init__(self, config):

            super().__init__(config)

            self.config = config
            self.wav2vec2 = Wav2Vec2Model(config)
            self.classifier = RegressionHead(config)
            self.init_weights()

        def forward(
                self,
                input_values,
        ):
            input_values = input_values
            outputs = self.wav2vec2(input_values)
            hidden_states = outputs[0]
            hidden_states = torch.mean(hidden_states, dim=1)
            logits = self.classifier(hidden_states)

            return hidden_states, logits
    # load model from hub
    model_name = 'audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim'
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = EmotionModel.from_pretrained(model_name)
    if torch.cuda.is_available():
        model = model.to(device)

    def process_func(
        x: np.ndarray,
        sampling_rate: int,
        embeddings: bool = False,
    ) -> np.ndarray:
        r"""Predict emotions or extract embeddings from raw audio signal."""

        # run through processor to normalize signal
        # always returns a batch, so we just get the first entry
        # then we put it on the device
        y = processor(x, sampling_rate=sampling_rate) 
        y = y['input_values'][0]
        y = y.reshape(1, -1)
        y = y
        y = torch.from_numpy(y).to(device)

        # run through model
        with torch.no_grad():
            y = model(y)[0 if embeddings else 1]

        # convert to numpy
        y = y.detach().cpu().numpy()

        return y
    
    audio_features = torch.empty((0,3), dtype = torch.float)
    # Iterate over the files in the video folder
    video_paths = os.listdir(video_folder)
    for video_file in video_paths:
        if video_file.endswith(".mp4"):
            # Construct the full path to the video file
            input_video_path = os.path.join(video_folder, video_file)

            # Convert mp4 to wav
            if constants.DATASET == 'MELD':
                DATA_DIR = constants.MELD_DATA_DIR
            elif constants.DATASET == 'C-EXPR-DB':
                DATA_DIR = constants.C_EXPR_DATA_DIR
            wav_file = f"{DATA_DIR}/temp_audio.mp3"

            # Extract audio from video and creates .wav file
            command = f"{constants.FFMPEG_PATH} -i {input_video_path} -ab 160k -ac 2 -ar 16000 -vn {wav_file}"
            p = subprocess.Popen(command, shell=True, stdin=subprocess.PIPE, stderr=subprocess.STDOUT, stdout=subprocess.PIPE, close_fds=True)
            out,err = p.communicate()

            audio_features = torch.cat((audio_features, extract_AVD(wav_file, process_func)))

            if detect_voice:
                voice_activity = extract_voice_actitity(wav_file, min_silence_duration_ms=200, threshold = 0.6)
                for record in voice_activity:
                    record['path'] = video_file
                voice_activity_df = pd.DataFrame.from_records(voice_activity)
                voice_activity_df.to_csv(f'{output_folder}/voice_activity/{video_file[:-4]}.csv')
                
            if os.path.exists(wav_file):
                os.remove(wav_file)
            print(f'Video {input_video_path} processed')
    audio_features_df = pd.DataFrame(audio_features, columns = ['Arousal', 'Dominance', 'Valence'])
    audio_features_df['path'] = video_paths
    audio_features_df.to_csv(f'{output_folder}/processed_audio_features.csv')

def extract_AVD(wav_file, process_func):
    """
    Computes Valence, Arousal and Dominance of the given audio file using a pre-trained wav2vec model.
    """

    if os.path.exists(wav_file):
        waveform, sr = librosa.load(wav_file, sr = 16000)
        avd = process_func(waveform, sr)
        features = torch.tensor(avd)
        
    else:
        print("Error: wav file not found")
        return torch.zeros((1,3))
        
    return features

def extract_voice_actitity(wav_path, min_speech_duration_ms=250, min_silence_duration_ms=100, threshold = 0.5, sampling_rate = 16000):
    """
    Extracts the timestamps of each voice activity and associated time stamps
    Args:
        wav_path (string): Path of the audio file
        min_speech_duration_ms (int): Final speech chunks shorter min_speech_duration_ms are thrown out
            Default: 250.
        min_silence_duration_ms (int): In the end of each speech chunk wait for min_silence_duration_ms before separating it
            Default: 100.
        threshold (float): Speech threshold. Silero VAD outputs speech probabilities for each audio chunk, probabilities ABOVE this value are considered as SPEECH.
        It is better to tune this parameter for each dataset separately, but "lazy" 0.5 is pretty good for most datasets.
            Default: 0.5.
        sampling_rate (int): Sampling rate for reading the audio
            Default: 16000.
    """

    model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                              model='silero_vad')

    (get_speech_timestamps,
    save_audio,
    read_audio,
    VADIterator,
    collect_chunks) = utils

    wav = read_audio(wav_path, sampling_rate = sampling_rate)
    speech_timestamps = get_speech_timestamps(wav, model, sampling_rate=sampling_rate, min_speech_duration_ms=min_speech_duration_ms,
                                               min_silence_duration_ms=min_silence_duration_ms, threshold = threshold)
    # load whisper model and processor
    processor = WhisperProcessor.from_pretrained("openai/whisper-large-v2")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v2")
    model.config.forced_decoder_ids = None

    output = []
    for speech_timestamp in speech_timestamps:
        start_time = speech_timestamp['start']
        end_time = speech_timestamp['end']
        
        input_features = processor(wav[start_time:end_time], sampling_rate=sampling_rate, return_tensors="pt").input_features 

        # generate token ids
        predicted_ids = model.generate(input_features)
        # decode token ids to text
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)

        output.append({
            'start': start_time/sampling_rate,
            'end': end_time/sampling_rate,
            'transcription':transcription})
    return output



def analyze_tone_from_videos(video_folder, output_dir):
    """
    Uses the Hume API Prosody Model to analyze the tone of each video.
    Args:
        video_folder (string): Path to folder containing the videos
        output_dir (string): Path to the directory in which the .csv files will be stored
    """
    video_paths = os.listdir(video_folder)
    for i, video_file in enumerate(video_paths):
        if video_file.endswith(".mp4") and not(os.path.exists(f'{output_dir}/{video_file[:-4]}.json')):
            video_path = f'{video_folder}/{video_file}'

            wav_path = f'{video_path[:-4]}.wav'
            # Extract audio from video and creates .wav file
            command = f"{constants.FFMPEG_PATH} -i {video_path} -ab 160k -ac 2 -ar 16000 -vn {wav_path}"
            p = subprocess.Popen(command, shell=True, stdin=subprocess.PIPE, stderr=subprocess.STDOUT, stdout=subprocess.PIPE, close_fds=True)
            out,err = p.communicate()

            if os.path.exists(wav_path):    
                client = HumeBatchClient(constants.API_KEY)

                prosody_config = ProsodyConfig()
                transcription_config = TranscriptionConfig(language='en')
                job = client.submit_job(None, [prosody_config], files=[wav_path], transcription_config=transcription_config)

                print("Running...", job)
                job.await_complete()
                print("Job completed with status: ", job.get_status())
                job.download_predictions(f"{output_dir}/{video_file[:-4]}.json")

            print(f"Predictions {i}/{len(video_paths)} downloaded to {output_dir}/{video_file[:-4]}.json")

def extract_hume_features(input_folder, output_folder):
    """
    Extracts the characteristics scores from the .json file produced by the Hume prosody model API and create a .csv file for each .json file.
    Args:
        input_folder (string): Path to the folder containing the .json files
        output_folder (string): Path to the output_folder
    """
    for file in os.listdir(input_folder):
        if file.endswith(".json") and not(os.path.isfile(f"{output_folder}/{file[:-5]}.csv")):
            with open(f"{input_folder}/{file}") as f:
                data = json.load(f)
                dfs = []
            if data[0]['results']['predictions'] == [] or data[0]['results']['predictions'][0]['models']['prosody']['grouped_predictions'] ==[]:
                features_df = pd.DataFrame(columns=['name','score','start','end'])
            else:
                for x in data:
                    for y in x['results']['predictions'][0]['models']['prosody']['grouped_predictions'][0]['predictions']:
                        df = pd.DataFrame.from_records(y['emotions']).sort_values('score', ascending=False)
                        df['start'], df['end']  = pd.Series(y['time'])
                        dfs.append(df)
                features_df = pd.concat(dfs)
            features_df.to_csv(f"{output_folder}/{file[:-5]}.csv")

