import os
import yaml
from pathlib import Path
from inspect import getsourcefile


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

COMPOUND_EMOTIONS = yaml.safe_load(Path(f'{FOLDS_PATH}/split-0/class_id.yaml').read_text())
COMPOUND_EMOTIONS['Other'] = 7
API_KEY = '' # Hume API key
LLAMA_TOKEN = '' # Hugging face Llama access token
FFMPEG_PATH = '/home/ens/AU58490/work/YP/ffmpeg-6.1-amd64-static/ffmpeg' # Path to ffmpeg