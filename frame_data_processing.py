
import numpy as np
import torch
from torch.utils.data import TensorDataset
import constants
import pandas as pd
from transformers import BertTokenizer, AutoTokenizer
import os
import re
import time
import random



def preprocess_data(device, args, tokenizer_name='bert', sample_k_prompt = None, return_sentences = False, return_val_df = False, specific_video_paths = None):
    """
    Preprocesses the data from MELD for training. returns a list of dictionnaries (one for each fold), split contains train/val/test tensor datasets
    Args:
        tokenizer_name (string): name of the tokenizer to use (same as the model name).
        sample_k_prompt (int): to sample randomly a given number of prompts
        return_sentences (boolean): if True then for each fold dictionnary d, d['test_sentences'] contains the sentences corresponding to the input_ids in d['test']
        return_val_df (boolean): if True then split_dict['val_df'] contains the validation dataframe
        specific_video_paths (str list): list of paths to specific videos to preprocess (only these videos will be preprocessed)
    """
    if args.dataset == 'MELD':
        #Read text datasets
        train = pd.read_csv(f'{constants.MELD_DATA_DIR}/train/train_sent_emo.csv', encoding='utf8')[['Utterance', 'Emotion', 'Dialogue_ID', 'Utterance_ID']]
        test = pd.read_csv(f'{constants.MELD_DATA_DIR}/test/test_sent_emo.csv')[['Utterance', 'Emotion', 'Dialogue_ID', 'Utterance_ID']]
        val = pd.read_csv(f'{constants.MELD_DATA_DIR}/dev/dev_sent_emo.csv')[['Utterance', 'Emotion', 'Dialogue_ID', 'Utterance_ID']]
        for df, split in zip([train, val, test], ['train', 'dev', 'test']):
            df['Emotion'] = [constants.EMOTIONS[x] for x in df['Emotion']]
            df['label'] = df['Emotion']
            indices, rowSeries = zip(*df.iterrows())
            df['path'] = [f'{constants.MELD_DATA_DIR}/{split}/videos/dia{x["Dialogue_ID"]}_utt{x["Utterance_ID"]}.mp4' for x in rowSeries]
            df['CE_path'] = [f'{constants.MELD_DATA_DIR}/{split}/videos/dia{x["Dialogue_ID"]}_utt{x["Utterance_ID"]}.mp4' for x in rowSeries]
        splits = ['full-split']

    elif args.dataset == 'C-EXPR-DB':
        compound_emotion = constants.COMPOUND_EMOTIONS
        annotation_df_list = []
        annotation_path = os.path.join(constants.C_EXPR_ANNOT_DIR, 'annotation')
        for annotation_file in os.listdir(annotation_path):
            if annotation_file.endswith('.csv'):
                annotation_df = pd.read_csv(os.path.join(annotation_path, annotation_file))
                annotation_df['path']= f'{constants.C_EXPR_DATA_DIR}/{annotation_file[:-4]}.mp4'
                labels = (annotation_df[list(compound_emotion)]==1).idxmax(axis=1)
                indexes = (annotation_df[list(compound_emotion)].fillna(0).cumsum()[(annotation_df[list(compound_emotion)]==1)].stack().astype(int)-1).tolist()
                annotation_df['trimmed_path'] = [f'{constants.C_EXPR_ANNOT_DIR}/trimmed_videos/{"-".join(x.split())}/{annotation_file[:-4]}_{"-".join(x.split())}_{index}.mp4' for x, index in zip(labels, indexes)]
                annotation_df_list.append(annotation_df)
        annotation_df = pd.concat(annotation_df_list)
        
        hume_features_df_list = []
        for file in os.listdir(f'{constants.C_EXPR_DATA_DIR}/hume_features'):
            hume_features_df = pd.read_csv(f'{constants.C_EXPR_DATA_DIR}/hume_features/{file}')
            hume_features_df['path'] = f'{constants.C_EXPR_DATA_DIR}/{file[:-4]}.mp4'
            hume_features_df_list.append(hume_features_df)
        hume = pd.concat(hume_features_df_list)

        transcription_df_list = []
        for file in os.listdir(f'{constants.C_EXPR_DATA_DIR}/voice_activity'):
            transcription_df = pd.read_csv(f'{constants.C_EXPR_DATA_DIR}/voice_activity/{file}')
            transcription_df['path'] = f'{constants.C_EXPR_DATA_DIR}/{file[:-4]}.mp4'
            transcription_df_list.append(transcription_df)
        transcription_df = pd.concat(transcription_df_list)
        transcription_df['transcription'] = transcription_df['transcription'].apply(lambda x: re.sub('[\[\]\\\'\"]', '',x)[1:])
        transcription_df = transcription_df.groupby(['path'], as_index=False)['transcription'].apply(lambda x: '\n '.join([y for y in x]))
        splits = os.listdir(constants.FOLDS_PATH)
        splits.sort()
        splits = [splits[i] for i in args.used_folds]
        audio_features_df = pd.read_csv(f'{constants.C_EXPR_DATA_DIR}/processed_audio_features.csv')
        audio_features_df['path'] = [f'{constants.C_EXPR_DATA_DIR}/{x}' for x in audio_features_df['path']]

    splits_dict_list = []
    for split in splits:
        if args.dataset == 'C-EXPR-DB':
            train, test, val = [pd.read_csv(f'{constants.C_EXPR_ANNOT_DIR}/folds/{split}/{x}.txt', names=['CE_path', 'label']) for x in ['train', 'test', 'val']]
        updated_dfs = []
        for df, split_name in zip([train, test, val], ['train', 'test', 'dev']):
            video_paths = df['CE_path']
            if specific_video_paths != None:
                video_paths =  specific_video_paths
            if args.dataset == 'MELD':
                audio_features_df = pd.read_csv(f'{constants.MELD_DATA_DIR}/{split_name}/processed_audio_features.csv')
                audio_features_df['path'] = [f'{constants.MELD_DATA_DIR}/{split_name}/videos/{x}' for x in audio_features_df['path']]
                
                hume_features_df_list = []
                for file in os.listdir(f'{constants.MELD_DATA_DIR}/{split_name}/hume_features'):
                    hume_features_df = pd.read_csv(f'{constants.MELD_DATA_DIR}/{split_name}/hume_features/{file}')
                    hume_features_df['path'] = f'{constants.MELD_DATA_DIR}/{split_name}/videos/{file[:-4]}.mp4'
                    hume_features_df_list.append(hume_features_df)
                hume = pd.concat(hume_features_df_list)
            
            
                if args.use_meld_train_subset and split_name == 'train':
                    video_paths = pd.read_csv(constants.MELD_TRAIN_SUBSET_PATH, names=['paths'])
                    video_paths = [x.split("/")[-1] for x in video_paths['paths']]
                    video_paths = [f'{constants.MELD_DATA_DIR}/train/videos/{x}.mp4' for x in video_paths]
            
            video_features_df_list = []
                
            for video_path in video_paths:
                if args.dataset == 'MELD':
                    video_path = video_path.split("/")[-1]
                    if os.path.exists(f'{constants.MELD_DATA_DIR}/{split_name}/video_features/{video_path[:-4]}.csv'):
                        features_df = pd.read_csv(f'{constants.MELD_DATA_DIR}/{split_name}/video_features/{video_path[:-4]}.csv').drop('Unnamed: 0', axis=1)
                    else:
                        continue
                elif args.dataset == 'C-EXPR-DB':
                    features_df = pd.read_csv(f'{constants.C_EXPR_DATA_DIR}/video_features/{video_path}.csv').drop('Unnamed: 0', axis=1)
                
                predominant_face_idx = features_df[['frame', 'FaceScore']].groupby('frame')['FaceScore'].transform('max') == features_df['FaceScore']
                features_df = features_df[predominant_face_idx]
                video_features_df_list.append(features_df)
            video_features_df = pd.concat(video_features_df_list)

            summarized_video_features_df = pd.DataFrame(columns=list(constants.ACTION_UNITS.keys())+['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise', 'neutral', 'path', 'frame'])
            for video in video_features_df['input'].unique():
                idx = video_features_df['input']== video
                input_frames = video_features_df[idx].set_index('frame')
                
                if args.training_method == 'windows' and (split_name == 'train' or (not(args.use_single_frame_per_video) and args.dataset=='MELD')):
                    interval_ids = [i for i in range(0,len(input_frames.index),args.window_hop)] + [len(input_frames.index)]
                    for idz in range(len(interval_ids)-1):
                        context = input_frames.iloc[interval_ids[idz]:interval_ids[idz+1]]
                        action_units = pd.DataFrame([context[list(constants.ACTION_UNITS.keys())+['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise', 'neutral']].max(axis=0)])
                        action_units['CE_path'] = video
                        action_units['frame'] = interval_ids[idz]
                        summarized_video_features_df = pd.concat([summarized_video_features_df, action_units], ignore_index=True)

                elif args.training_method == 'all' or split_name != 'train':
                    window_size = args.window_size
                    for frame in input_frames.index:
                        
                        if args.use_single_frame_per_video and args.dataset == 'MELD':
                            window_size = len(input_frames)*10
                        first_context_frame = max((frame-window_size//2),0)
                        last_context_frame = min(frame+window_size//2+1, len(input_frames))
                        context = input_frames.iloc[first_context_frame:last_context_frame]
                        action_units = pd.DataFrame([context[list(constants.ACTION_UNITS.keys())+['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise', 'neutral']].max(axis=0)])
                        action_units['CE_path'] = video
                        action_units['frame'] = frame
                        summarized_video_features_df = pd.concat([summarized_video_features_df, action_units], ignore_index=True)
                        if args.use_single_frame_per_video and args.dataset == 'MELD':
                            break
                        
            if args.dataset == 'C-EXPR-DB':
                df['CE_path'] = [f'{constants.C_EXPR_ANNOT_DIR}/trimmed_videos/{x}.mp4' for x in df['CE_path']]
                df = pd.merge(df, annotation_df.set_index('trimmed_path'), how='left', left_on=['CE_path'], right_on=['trimmed_path'], suffixes=('_left', ''))
            df = pd.merge(summarized_video_features_df, df.set_index('CE_path'), how = 'left', left_on=['CE_path'], right_on=['CE_path'], suffixes= ['_left', ''])
            if sample_k_prompt != None:
                df = df.sample(sample_k_prompt)
            df = pd.merge(df, audio_features_df, how='left', left_on='path', right_on='path')
            if args.dataset == 'C-EXPR-DB':
                df['Begin Time - hh:mm:ss.ms'] = df['Begin Time - hh:mm:ss.ms'].apply(time_to_seconds)
                df['End Time - hh:mm:ss.ms'] = df['End Time - hh:mm:ss.ms'].apply(time_to_seconds)

            #add hume_features based on context
            context_features_list = []
            if args.dataset == 'C-EXPR-DB':
                for clip in df['CE_path'].unique():
                    idx_label = df['CE_path']==clip                
                    start = df[idx_label]['Begin Time - hh:mm:ss.ms'].iloc[0]
                    end = df[idx_label]['End Time - hh:mm:ss.ms'].iloc[0]
                    idx_clip_hume = hume['path']==df[idx_label].iloc[0]['path']
                    idx_utt = ((hume[idx_clip_hume]['start'] <= end+args.hume_supp_context) & (hume[idx_clip_hume]['start'] >= start-args.hume_supp_context))  | ((hume[idx_clip_hume]['start'] <= end+args.hume_supp_context) & (hume[idx_clip_hume]['end']>=start-args.hume_supp_context))
                    
                    context_hume_features = hume[idx_clip_hume][idx_utt].nlargest(10, 'score')
                        
                    context_hume_features['CE_path'] =clip
                    context_features_list.append(context_hume_features)
            elif args.dataset == 'MELD':
                for clip in hume['path'].unique():
                    idx_label = hume['path']==clip                
                    context_hume_features = hume[idx_label].nlargest(10, 'score')
                        
                    context_hume_features['CE_path'] =clip
                    context_features_list.append(context_hume_features)
            context_hume_features_df =pd.concat(context_features_list)
            context_hume_features_df = context_hume_features_df.groupby(['path', 'CE_path'], as_index=False).apply(concatenate_hume_features, include_groups=False)
            context_hume_features_df.columns = ['path', 'CE_path', 'characteristics']
            df = pd.merge(df, context_hume_features_df, how='left', left_on=['path', 'CE_path'], right_on=['path', 'CE_path'])
            if args.dataset == 'C-EXPR-DB':
                if args.transcription_context == "clip":
                    #add speech transcription corresponding to the clip
                    transcription_df = pd.read_csv(f'{constants.C_EXPR_DATA_DIR}/clips_transcription.csv', names = ['CE_path', 'transcription'])
                    transcription_df['CE_path'] = [f'{constants.C_EXPR_ANNOT_DIR}/trimmed_videos/{x}.mp4' for x in transcription_df['CE_path']]
                    df = pd.merge(df, transcription_df, how='left', left_on='CE_path', right_on='CE_path')
                else :
                    #add speech transcription of the whole clip
                    df = pd.merge(df, transcription_df, how='left', left_on='path', right_on='path')
                df['label'] = df['label'].astype(int)
                if not(args.use_other_class) or split_name != 'train':
                    df = df[df['label'] != 7].reset_index()
            df['Utterance'] = df.apply(lambda x: apply_prompt_template(x, args=args), axis=1)
            
            
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
    max_len = 0
    for split_dict in splits_dict_list:
        train, test, val = split_dict['train'], split_dict['test'], split_dict['val']
        
        sentences = np.concatenate([np.array(train['Utterance']), np.array(test['Utterance']), np.array(val['Utterance'])])
        for sentence in sentences:
            input_ids=tokenizer.encode(sentence, add_special_tokens=True)
            max_len = max(max_len, len(input_ids))
    print(f'Max sentence length: {max_len}')
    print(f'shape of train dataframe : {split_dict["train"].shape}')
    print(f'shape of val dataframe : {split_dict["val"].shape}')
    print(f'number of training emotions : {len(split_dict["train"]["label"].unique())}')
    print(f'number of validation emotions : {len(split_dict["val"]["label"].unique())}')
    time.sleep(7)
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
            labels = torch.tensor(list(split_dict[sub_split]['label']))
            dataset = TensorDataset(input_ids, attention_masks, labels)
            datasets.append(dataset)
        if return_sentences:
            split_dict['test_sentences'] = list(split_dict['test']['Utterance'])
        if return_val_df:
            split_dict['val_df'] = split_dict['val']
        split_dict['train'] = datasets[0]
        split_dict['test'] = datasets[1]
        split_dict['val'] = datasets[2]
        

    return splits_dict_list



def apply_prompt_template(row, args):
    """
    Applies prompt template to the given row
    Args:
        row (pd.Series): pd.Series containing the Utterance, the Arousal, Dominance and Valence values
        textualize_audio_levels (boolean): if True then levels of arousal, valence and dominance are textualized into 'high' or 'low'
    """
    prompt = ''
    aus = row[list(constants.ACTION_UNITS.keys())]
    activated_aus = ''.join([f'{constants.ACTION_UNITS[x]}, ' if aus[x]>0.5 else '' for x in constants.ACTION_UNITS.keys()])
    
    tone_description = row['characteristics']
    if not(isinstance(tone_description, str)):
        tone_description = 'No speech detected'
    if args.dataset == 'MELD':
        transcription = row['Utterance']
    elif args.dataset == 'C-EXPR-DB':
        transcription = row['transcription']
    visual_emotions = row[['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise', 'neutral']].astype(float)
    visual_emotions_text = ', '.join(visual_emotions.nlargest(n=3).index if visual_emotions.nlargest(1).iloc[0]>0 else [])
    if len(activated_aus)>2:
        activated_aus = activated_aus[:-2]
    else:
        activated_aus = 'No face detected'
        visual_emotions_text = 'No face detected'
    valence = round(row['Valence'], 2) 
    arousal = round(row['Arousal'], 2)
    dominance = round(row['Dominance'], 2)
    
    if 'T' in args.used_modalities:
        prompt+=f"Speech transcription of the video : {transcription}; "
    if 'V' in args.used_modalities:
        prompt += f"Facial Action Units activated during the video : {activated_aus}; Emotions predicted from visual modality: {visual_emotions_text}; "
    
    if 'A' in args.used_modalities:
        prompt += f"Characteristics of the prosody : {tone_description}; "
        if args.textualize_audio_levels:
            feature_levels = []
            for feature in [valence, arousal, dominance]:
                feature_levels.append('High' if feature>=0.5 else 'Low')
            prompt += f' Audio emotional state :  {feature_levels[0]} Valence, {feature_levels[1]} Arousal, {feature_levels[2]} Dominance'
        else:
            prompt += f" Audio emotional state : Arousal level (between 0 and 1) of {arousal}, Valence level (between 0 and 1) of {valence}, Dominance level (between 0 and 1) of {dominance}"

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