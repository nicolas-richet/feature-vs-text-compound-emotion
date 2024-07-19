

    
import numpy as np
import torch
from torch.utils.data import TensorDataset
import constants
import pandas as pd
from transformers import BertTokenizer, AutoTokenizer
import subprocess
import os
import re
import torch.nn as nn
from transformers import Wav2Vec2Processor, WhisperProcessor, WhisperForConditionalGeneration
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
)
import librosa
from feat import Detector




def preprocess_data(device, tokenizer_name='bert', zeroshot =False):
    """
    Preprocesses the data from MELD for training. returns train/dev/test TensorDatasets of tokenized utterances and the class weights. If zeroshot is True then simply returns test pandas dataset
    Args:
        tokenizer_name (string): name of the tokenizer to use (same as the model name).
        zeroshot (boolean): if True, the model is considered a Causal LM and will be used for zero-shot classification.
            Default: False.
    """
    if constants.DATASET == 'MELD':
        #Read text datasets
        train = pd.read_csv(f'{constants.DATA_DIR}/train/train_sent_emo.csv', encoding='utf8')[['Utterance', 'Emotion', 'Dialogue_ID', 'Utterance_ID']]
        test = pd.read_csv(f'{constants.DATA_DIR}/test/test_sent_emo.csv')[['Utterance', 'Emotion', 'Dialogue_ID', 'Utterance_ID']]
        val = pd.read_csv(f'{constants.DATA_DIR}/dev/dev_sent_emo.csv')[['Utterance', 'Emotion', 'Dialogue_ID', 'Utterance_ID']]
        for df, split in zip([train, val, test], ['train', 'dev', 'test']):
            df['Emotion'] = [constants.EMOTIONS[x] for x in df['Emotion']]
            df['label'] = df['Emotion']
            indices, rowSeries = zip(*df.iterrows())
            df['path'] = [f'{constants.DATA_DIR}/{split}/videos/dia{x["Dialogue_ID"]}_utt{x["Utterance_ID"]}.mp4' for x in rowSeries]
            df['CE_path'] = [f'{constants.DATA_DIR}/{split}/videos/dia{x["Dialogue_ID"]}_utt{x["Utterance_ID"]}.mp4' for x in rowSeries]
        splits = ['full-split']

    elif constants.DATASET == 'C-EXPR-DB':
        compound_emotion = constants.COMPOUND_EMOTIONS
        annotation_df_list = []
        annotation_path = os.path.join(constants.DATASET_DIR, 'annotation')
        for annotation_file in os.listdir(annotation_path):
            if annotation_file.endswith('.csv'):
                annotation_df = pd.read_csv(os.path.join(annotation_path, annotation_file))
                annotation_df['path']= f'{constants.DATA_DIR}/{annotation_file[:-4]}.mp4'
                labels = (annotation_df[list(compound_emotion)]==1).idxmax(axis=1)
                indexes = (annotation_df[list(compound_emotion)].fillna(0).cumsum()[(annotation_df[list(compound_emotion)]==1)].stack().astype(int)-1).tolist()
                annotation_df['trimmed_path'] = [f'{constants.DATASET_DIR}/trimmed_videos/{"-".join(x.split())}/{annotation_file[:-4]}_{"-".join(x.split())}_{index}.mp4' for x, index in zip(labels, indexes)]
                annotation_df_list.append(annotation_df)
        annotation_df = pd.concat(annotation_df_list)
        
    if constants.DATASET == 'C-EXPR-DB':
        hume_features_df_list = []
        for file in os.listdir(f'{constants.DATA_DIR}/hume_features'):
            hume_features_df = pd.read_csv(f'{constants.DATA_DIR}/hume_features/{file}')
            hume_features_df['path'] = f'{constants.DATA_DIR}/{file[:-4]}.mp4'
            hume_features_df_list.append(hume_features_df)
        hume = pd.concat(hume_features_df_list)

    if constants.DATASET == 'C-EXPR-DB':
        transcription_df_list = []
        for file in os.listdir(f'{constants.DATA_DIR}/voice_activity'):
            transcription_df = pd.read_csv(f'{constants.DATA_DIR}/voice_activity/{file}')
            transcription_df['path'] = f'{constants.DATA_DIR}/{file[:-4]}.mp4'
            transcription_df_list.append(transcription_df)
        transcription_df = pd.concat(transcription_df_list)
        transcription_df['transcription'] = transcription_df['transcription'].apply(lambda x: re.sub('[\[\]\\\'\"]', '',x)[1:])
        transcription_df = transcription_df.groupby(['path'], as_index=False)['transcription'].apply(lambda x: '\n '.join([y for y in x]))
        splits = os.listdir(constants.FOLDS_PATH)
        audio_features_df = pd.read_csv(f'{constants.DATA_DIR}/processed_audio_features.csv')
        audio_features_df['path'] = [f'{constants.DATA_DIR}/{x}' for x in audio_features_df['path']]

    splits_dict_list = []
    for split in splits:
        if constants.DATASET == 'C-EXPR-DB':
            train, test, val = [pd.read_csv(f'{constants.DATASET_DIR}/folds/{split}/{x}.txt', names=['CE_path', 'label']) for x in ['train', 'test', 'val']]
        updated_dfs = []
        for df, split_name in zip([train, test, val], ['train', 'test', 'dev']):
            if constants.DATASET == 'MELD':
                audio_features_df = pd.read_csv(f'{constants.DATA_DIR}/{split_name}/processed_audio_features.csv')
                audio_features_df['path'] = [f'{constants.DATA_DIR}/{split_name}/videos/{x}' for x in audio_features_df['path']]
                
                if split_name == 'test':
                    updated_dfs.append(df)
                    continue
                hume_features_df_list = []
                for file in os.listdir(f'{constants.DATA_DIR}/{split_name}/hume_features'):
                    hume_features_df = pd.read_csv(f'{constants.DATA_DIR}/{split_name}/hume_features/{file}')
                    hume_features_df['path'] = f'{constants.DATA_DIR}/{split_name}/videos/{file[:-4]}.mp4'
                    hume_features_df_list.append(hume_features_df)
                hume = pd.concat(hume_features_df_list)
            
            video_paths = df['CE_path']
            if constants.USE_TRAIN_SUBSET and split_name == 'train':
                video_paths = pd.read_csv(constants.TRAIN_SUBSET_PATH, names=['paths'])
                video_paths = [x.split("/")[-1] for x in video_paths['paths']]
                video_paths = [f'{constants.DATA_DIR}/train/videos/{x}.mp4' for x in video_paths]
            
            video_features_df_list = []
            for video_path in video_paths:
                if constants.DATASET == 'MELD':
                    video_path = video_path.split("/")[-1]
                    if os.path.exists(f'{constants.DATA_DIR}/{split_name}/video_features/{video_path[:-4]}.csv'):
                        features_df = pd.read_csv(f'{constants.DATA_DIR}/{split_name}/video_features/{video_path[:-4]}.csv').drop('Unnamed: 0', axis=1)
                    else:
                        continue
                elif constants.DATASET == 'C-EXPR-DB':
                    features_df = pd.read_csv(f'{constants.DATA_DIR}/video_features/{video_path}.csv').drop('Unnamed: 0', axis=1)
                
                predominant_face_idx = features_df[['frame', 'FaceScore']].groupby('frame')['FaceScore'].transform('max') == features_df['FaceScore']
                features_df = features_df[predominant_face_idx]
                video_features_df_list.append(features_df)
            video_features_df = pd.concat(video_features_df_list)

            summarized_video_features_df = pd.DataFrame(columns=list(constants.ACTION_UNITS.keys())+['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise', 'neutral', 'path', 'frame', 'approx_time'])
            for video in video_features_df['input'].unique():
                idx = video_features_df['input']== video
                input_frames = video_features_df[idx].set_index('frame')
                for frame in input_frames.index:
                    first_context_frame = max((frame-constants.WINDOW_SIZE//2),0)
                    last_context_frame = min(frame+constants.WINDOW_SIZE//2+1, len(input_frames))
                    context = input_frames.iloc[first_context_frame:last_context_frame]
                    action_units = pd.DataFrame([context[list(constants.ACTION_UNITS.keys())+['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise', 'neutral']].max(axis=0)])
                    action_units['CE_path'] = video
                    action_units['frame'] = frame
                    action_units['approx_time'] = input_frames.iloc[frame]['approx_time']
                    summarized_video_features_df = pd.concat([summarized_video_features_df, action_units], ignore_index=True)
                    if split_name == 'dev':
                        break
                    if constants.USE_SINGLE_FRAME_PER_VIDEO and split_name == 'train':
                        break
            if constants.DATASET == 'C-EXPR-DB':
                df['CE_path'] = [f'{constants.DATASET_DIR}/trimmed_videos/{x}.mp4' for x in df['CE_path']]
                df = pd.merge(df, annotation_df.set_index('trimmed_path'), how='left', left_on=['CE_path'], right_on=['trimmed_path'], suffixes=('_left', ''))
            df = pd.merge(summarized_video_features_df, df.set_index('CE_path'), how = 'left', left_on=['CE_path'], right_on=['CE_path'], suffixes= ['_left', ''])
            df = pd.merge(df, audio_features_df, how='left', left_on='path', right_on='path')
            if constants.DATASET == 'C-EXPR-DB':
                df['Begin Time - hh:mm:ss.ms'] = df['Begin Time - hh:mm:ss.ms'].apply(time_to_seconds)
                df['End Time - hh:mm:ss.ms'] = df['End Time - hh:mm:ss.ms'].apply(time_to_seconds)

            #add hume_features based on context
            context_features_list = []
            if constants.DATASET == 'C-EXPR-DB':
                for clip in df['CE_path'].unique():
                    idx_label = df['CE_path']==clip                
                    start = df[idx_label]['Begin Time - hh:mm:ss.ms'].iloc[0]
                    end = df[idx_label]['End Time - hh:mm:ss.ms'].iloc[0]
                    idx_clip_hume = hume['path']==df[idx_label].iloc[0]['path']
                    idx_utt = ((hume[idx_clip_hume]['start'] <= end+constants.HUME_SUPP_CONTEXT) & (hume[idx_clip_hume]['start'] >= start-constants.HUME_SUPP_CONTEXT))  | ((hume[idx_clip_hume]['start'] <= end+constants.HUME_SUPP_CONTEXT) & (hume[idx_clip_hume]['end']>=start-constants.HUME_SUPP_CONTEXT))
                    
                    context_hume_features = hume[idx_clip_hume][idx_utt].nlargest(10, 'score')
                        
                    context_hume_features['CE_path'] =clip
                    context_features_list.append(context_hume_features)
            elif constants.DATASET == 'MELD':
                for clip in hume['path'].unique():
                    idx_label = hume['path']==clip                
                    context_hume_features = hume[idx_label].nlargest(10, 'score')
                        
                    context_hume_features['CE_path'] =clip
                    context_features_list.append(context_hume_features)
            context_hume_features_df =pd.concat(context_features_list)
            context_hume_features_df = context_hume_features_df.groupby(['path', 'CE_path'], as_index=False).apply(concatenate_hume_features, include_groups=False)
            context_hume_features_df.columns = ['path', 'CE_path', 'characteristics']
            df = pd.merge(df, context_hume_features_df, how='left', left_on=['path', 'CE_path'], right_on=['path', 'CE_path'])
            
            if constants.DATASET == 'C-EXPR-DB':
                if constants.TRANSCRIPTION_CONTEXT == "clip":
                    #add speech transcription corresponding to the clip
                    transcription_df = pd.read_csv(f'{constants.DATA_DIR}/clips_transcription.csv', names = ['CE_path', 'transcription'])
                    transcription_df['CE_path'] = [f'{constants.DATASET_DIR}/trimmed_videos/{x}.mp4' for x in transcription_df['CE_path']]
                    df = pd.merge(df, transcription_df, how='left', left_on='CE_path', right_on='CE_path')
                else :
                    #add speech transcription of the whole clip
                    df = pd.merge(df, transcription_df, how='left', left_on='path', right_on='path')
                df['label'] = df['label'].astype(int)
            df['Utterance'] = df.apply(apply_prompt_template, axis=1)
            
            updated_dfs.append(df)
        train, test, val = updated_dfs
        splits_dict_list.append({
            'split':split,
            'train':train,
            'test':test,
            'val':val
        })

    #Load Pretrainedtokenizer
    if tokenizer_name == 'bert':
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    elif tokenizer_name == 'roberta':
        #tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-emotion")
        tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-large-emotion-latest")
    elif tokenizer_name == 'llama2':
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", token = constants.LLAMA_TOKEN)
        tokenizer.pad_token = tokenizer.eos_token
    elif tokenizer_name == 'llama3':
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B", token = constants.LLAMA_TOKEN)
        tokenizer.pad_token = tokenizer.eos_token
    elif tokenizer_name == 'phi-2':
        tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")
        tokenizer.pad_token = tokenizer.eos_token

    elif tokenizer_name == 'phi-3':
        tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
        tokenizer.pad_token = tokenizer.eos_token

    # computes Max sequence length
    split_dict = splits_dict_list[0]
    train, test, val = split_dict['train'], split_dict['test'], split_dict['val']
    max_len = 0
    sentences = np.concatenate([np.array(train['Utterance']), np.array(test['Utterance']), np.array(val['Utterance'])])
    for sentence in sentences:
        input_ids=tokenizer.encode(sentence, add_special_tokens=True)
        max_len = max(max_len, len(input_ids))
    print(f'Max sentence length: {max_len}')
    print(f'shape of train dataframe : {split_dict["train"].shape}')
    for split_dict in splits_dict_list:
        datasets = []
        for sub_split in ['train', 'test', 'val']:
            input_ids = []
            attention_masks = []
            for utterance in split_dict[sub_split]['Utterance']:
                encoded_dict = tokenizer.encode_plus(utterance, add_special_tokens=True, max_length=max_len, padding = 'max_length', return_attention_mask=True, return_tensors='pt')
                input_ids.append(encoded_dict['input_ids'])
                attention_masks.append(encoded_dict['attention_mask'])
            input_ids = torch.cat(input_ids, dim=0)
            attention_masks = torch.cat(attention_masks, dim=0)
            labels = torch.tensor(split_dict[sub_split]['label'])
            dataset = TensorDataset(input_ids, attention_masks, labels)
            datasets.append(dataset)
        split_dict['train'] = datasets[0]
        split_dict['test'] = datasets[1]
        split_dict['val'] = datasets[2]

    return splits_dict_list

def preprocess_videos(video_folder, skip_frames, output_folder):
    """
    Applies the preprocessing to all videos in the folder. Action Units are computed for each video, then the max value (over time) for each AU is kept.

    """
    video_paths = os.listdir(video_folder)
    video_df_list = []
    print(f'Processing videos in {video_folder}')
    for i, video_file in enumerate(video_paths):
        if video_file.endswith(".mp4") and not(os.path.isfile(f'{output_folder}/{video_file[:-4]}_feat.csv')):
            # Construct the full path to the video file
            input_video_path = os.path.join(video_folder, video_file)
            detector = Detector()
            video_prediction = detector.detect_video(input_video_path, skip_frames = skip_frames)
            video_prediction = video_prediction.reset_index(drop=True).fillna(0)
            columns_kept = ['FaceRectX', 'FaceRectY', 'FaceRectWidth', 'FaceRectHeight', 'FaceScore'] + [x for x in constants.ACTION_UNITS.keys()] + ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise', 'neutral', 'input', 'frame', 'approx_time']
            video_df_list.append(video_prediction[columns_kept])
            video_prediction[columns_kept].to_csv(f'{output_folder}/{video_file[:-4]}_feat.csv')
        print(f'File {i}/{len(video_paths)} processed')
        #video_features_df = pd.concat(video_df_list)
    #return video_features_df


def preprocess_audios(video_folder, device, detect_voice = False):
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
            wav_file = f"{constants.DATA_DIR}/temp_audio.mp3"

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
                voice_activity_df.to_csv(f'{input_video_path[:-4]}_voice_activity.csv')
                
            if os.path.exists(wav_file):
                os.remove(wav_file)
            print(f'Video {input_video_path} processed')
    audio_features_df = pd.DataFrame(audio_features, columns = ['Arousal', 'Dominance', 'Valence'])
    audio_features_df['path'] = video_paths

    return audio_features_df

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

def apply_prompt_template(row):
    """
    Applies prompt template to the given row
    Args:
        row (pd.Series): pd.Series containing the Utterance, the Arousal, Dominance and Valence values
        textualize_audio_levels (boolean): if True then levels of arousal, valence and dominance are textualized into 'high' or 'low'
    """
    aus = row[list(constants.ACTION_UNITS.keys())]
    activated_aus = ''.join([f'{constants.ACTION_UNITS[x]}, ' if aus[x]>0.5 else '' for x in constants.ACTION_UNITS.keys()])
    activated_aus = activated_aus[:-2]
    tone_description = row['characteristics']
    if constants.DATASET == 'MELD':
        transcription = row['Utterance']
    elif constants.DATASET == 'C-EXPR-DB':
        transcription = row['transcription']
    visual_emotions = row[['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise', 'neutral']].astype(float)
    visual_emotions_text = ', '.join(visual_emotions.nlargest(n=3).index if visual_emotions.nlargest(1).iloc[0]>0 else [])
    valence = round(row['Valence'], 2) 
    arousal = round(row['Arousal'], 2)
    dominance = round(row['Dominance'], 2)
    prompt = f"Speech transcription of the video : \"{transcription}\"; Facial Action Units activated during the video : {activated_aus}; Emotions predicted from visual modality: {visual_emotions_text}; Characteristics of the prosody : {tone_description};"
    if constants.TEXTUALIZE_AUDIO_LEVELS:
        feature_levels = []
        for feature in [valence, arousal, dominance]:
            feature_levels.append('High' if feature>=0.5 else 'Low')
        prompt += f'Audio emotional state :  {feature_levels[0]} Valence, {feature_levels[1]} Arousal, {feature_levels[2]} Dominance'
    else:
        prompt += f"Audio emotional state : Arousal level (between 0 and 1) of {arousal}, Valence level (between 0 and 1) of {valence}, Dominance level (between 0 and 1) of {dominance}"
   
    return prompt

def time_to_seconds(time_str):
    """ 
    Split the string by colon and dot to separate hours, minutes, seconds, and milliseconds
    """
    parts = time_str.split(":")
    hours = int(parts[0])
    minutes = int(parts[1])
    seconds_and_ms = parts[2].split('.')
    seconds = int(seconds_and_ms[0])
    milliseconds = int(seconds_and_ms[1])
    
    total_seconds = hours * 3600 + minutes * 60 + seconds + milliseconds / 1000
    
    return total_seconds

def concatenate_hume_features(values):
    return ", ".join((f'High {y}' if x>0.3 else f'Low {y}') for x,y in zip(values['score'], values['name']))