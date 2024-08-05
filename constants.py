import os
import yaml
from pathlib import Path
from inspect import getsourcefile


# TRAINING PARAMETERS
PARAM_GRID = {
    'batch_size': [4,6, 8],#
    'lr': [7e-3, 3e-3, 9e-4],#5e-4, 7e-4, 9e-4,#[ 1e-6, 5e-6, 1e-5, 4e-5],#3e-2, 
    'weight_decay': 0.001
}
EPOCHS = 100
LORA_R = 8#64
LORA_ALPHA = 8#16
LORA_DROPOUT = 0.05
MAX_GRAD_NORM = 0.1#0.3
SAMPLES_PER_EPOCHS = 2000
GPU_ID = 1
TRANSCRIPTION_CONTEXT = "clip"  #context to use for the text modality. available : "clip", "video".
                                # Default : "video"
                                #"clip" uses the transcription of all utterances coinciding with the clip (as in any utterances starting/ending during the clip). 
                                #"video" uses the transcription of the whole video for each clip of the video.
MAX_PATIENCE = 50 # Number of epochs in a row with no improvement (on the validation set) the training should wait before stopping, use np.inf for no early stopping
TEXTUALIZE_AUDIO_LEVELS = True #Should the audio feature scores be textualized in the prompt (High/Low)
SKIPPED_FRAMES = None #Number of frame to skip during action unit pre-processing, use None to process every frames
USED_FOLDS = [0] #for all folds: [fold for fold in range(5)]
# Available Models: 'llama3', 'llama2', 'bert', 'roberta' 
# Zeroshot models ignore PARAM_GRID and EPOCHS
MODELS = ['llama3'] #List of model (by name) to try 
SEEDS = [0]
DATASET = 'C-EXPR-DB'
TRAINING_METHOD = 'windows' #'windows' for training with consecutives non over-lapping windows (window size must be defined)
                             #'all' to train using every frame (window size must be defined)
USE_OTHER_CLASS = False # Is the 'Other' class used in training (never used in validation). for C-EXPR-DB only.
WINDOW_SIZE = 20 # frames are used in training with windows
WINDOW_HOP = 10
USE_MELD_TRAIN_SUBSET = False

USE_SINGLE_FRAME_PER_VIDEO = True #for MELD only
USED_MODALITIES = ['T','V', 'A'] #['T', 'V', 'A']

MELD_DATA_DIR = '/home/ens/AU58490/work/YP/data' # Path to the folder containing the processed train/test/dev features folders
C_EXPR_DATA_DIR = '/home/ens/AU58490/work/YP/videos' # Path to the folder containing the processed features
C_EXPR_ANNOT_DIR = '/datasets/C-EXPR-DB' # Path to the folder containing the annotation folder
FOLDS_PATH = '/datasets/C-EXPR-DB/folds/' # Path to the folder containing the folds (with train/val/test splits)
MELD_TRAIN_SUBSET_PATH = '/home/ens/AU58490/work/YP/data/train/train-1.0.txt'

OUTPUT_PATH = '/home/ens/AU58490/work/YP/text-emotion/ABAW/outputs' # Path to the output folder
MODEL_PATH = '/home/ens/AU58490/work/YP/text-emotion/models' #Path to the folder where the model will be stored


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

HUME_SUPP_CONTEXT = 2
COMPOUND_EMOTIONS = yaml.safe_load(Path(f'{FOLDS_PATH}/split-0/class_id.yaml').read_text())
COMPOUND_EMOTIONS['Other'] = 7
API_KEY = 'PEDA47Gx7ViNR69OxxeaG9SDt3aAsvYstZkEkAtr810UHule' # Hume API key
LLAMA_TOKEN = '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' # Hugging face Llama access token
FFMPEG_PATH = '/home/ens/AU58490/work/YP/ffmpeg-6.1-amd64-static/ffmpeg' # Path to ffmpeg