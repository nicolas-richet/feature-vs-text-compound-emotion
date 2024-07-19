import os
import yaml
from pathlib import Path
from inspect import getsourcefile
DIR = '/home/ens/AU58490/work/YP'

# TRAINING PARAMETERS
PARAM_GRID = {
    'batch_size': [10],
    'lr': [7e-3],#5e-4, 7e-4, 9e-4,#[ 1e-6, 5e-6, 1e-5, 4e-5],#3e-2, 
    'weight_decay': 0.001
}
EPOCHS = 100
LORA_R = 8#64
LORA_ALPHA = 8#16
LORA_DROPOUT = 0.05
MAX_GRAD_NORM = 0.3
SAMPLES_PER_EPOCHS = 2000
GPU_ID = 2
TRANSCRIPTION_CONTEXT = "clip"  #context to use for the text modality. available : "clip", "video".
                                # Default : "video"
                                #"clip" uses the transcription of all utterances coinciding with the clip (as in any utterances starting/ending during the clip). 
                                #"video" uses the transcription of the whole video for each clip of the video.
MAX_PATIENCE = 35 # Number of epochs in a row with no improvement (on the validation set) the training should wait before stopping, use np.inf for no early stopping
TEXTUALIZE_AUDIO_LEVELS = True #Should the audio feature scores be textualized in the prompt (High/Low)
SKIPPED_FRAMES = None #Number of frame to skip during action unit pre-processing, use None to process every frames

# Available Models: 'bert', 'roberta', 'llama2', 'llama3', 'phi2', 'phi-3-zeroshot', 'stablelm-zeroshot'
# Zeroshot models ignore PARAM_GRID and EPOCHS
MODELS = ['llama3'] #List of model (by name) to try 
SEEDS = [0]
DATASET = 'MELD'
LLAMA_TOKEN =#hugging face llama token
OUTPUT_PATH = '/home/ens/AU58490/work/YP/audio_outputs'
UNIMODAL_TEXT_OUTPUT_PATH = '/home/ens/AU58490/work/YP/text-emotion/ABAW/outputs'
FFMPEG_PATH = '/home/ens/AU58490/work/YP/ffmpeg-6.1-amd64-static/ffmpeg'
MODEL_PATH = '/home/ens/AU58490/work/YP/text-emotion/models'
#DATA_DIR = '/home/ens/AU58490/work/YP/videos'
DATA_DIR = '/home/ens/AU58490/work/YP/data'
#DATASET_DIR = '/datasets/C-EXPR-DB'
DATASET_DIR = '/home/ens/AU58490/work/YP/data'
FOLDS_PATH = '/datasets/C-EXPR-DB/folds/'
ACTION_UNITS = {
    'AU01': 'Inner Brow Raiser',
    'AU02': 'Outer Brow Raiser',
    'AU04': 'Brow Lowerer',
    'AU05': 'Upper Lid Raiser',
    'AU06': 'Cheek Raiser',
    'AU07': 'Lid Tightener',
    'AU09': 'Nose Wrinkler',
    'AU10': 'Upper Lip Raiser',
    'AU11': 'Nasolabial Deepener',
    'AU12': 'Lip Corner Puller',
    'AU14': 'Dimpler',
    'AU15': 'Lip Corner Depressor',
    'AU17': 'Chin Raiser',
    'AU20': 'Lip Stretcher',
    'AU23': 'Lip Tightener',
    'AU24': 'Lip Pressor',
    'AU25': 'Lip Part',
    'AU26': 'Jaw Drop',
    'AU28': 'Lip Suck',
    'AU43': 'Eyes Closed'
}
EMOTIONS = {'anger':0,
            'disgust':1,
            'fear':2,
            'joy':3,
            'neutral':4,
            'sadness':5,
            'surprise':6}
COMPOUND_PAIRS = {
    '0': (2,6),
    '1': (3,6),
    '2': (5,6),
    '3': (1,6),
    '4': (0,6),
    '5': (2,5),
    '6': (0,5)
}
WINDOW_SIZE = 5
HUME_SUPP_CONTEXT = 2
COMPOUND_EMOTIONS = yaml.safe_load(Path('/datasets/C-EXPR-DB/folds/split-0/class_id.yaml').read_text())
COMPOUND_EMOTIONS['Other'] = 7
USE_TRAIN_SUBSET = False
TRAIN_SUBSET_PATH = '/home/ens/AU58490/work/YP/data/train/train-1.0.txt'
USE_SINGLE_FRAME_PER_VIDEO = True