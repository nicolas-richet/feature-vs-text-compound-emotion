import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Training Parameters")

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=3e-3, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.001, help='Weight decay value')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train for')
    parser.add_argument('--lora_r', type=int, default=8, help='LORA R parameter')
    parser.add_argument('--lora_alpha', type=int, default=8, help='LORA alpha parameter')
    parser.add_argument('--lora_dropout', type=float, default=0.05, help='LORA dropout rate')
    parser.add_argument('--max_grad_norm', type=float, default=0.3, help='Max gradient norm')
    parser.add_argument('--gpu_id', type=int, default=1, help='GPU ID to use')
    parser.add_argument('--transcription_context', type=str, default="clip", choices=["clip", "video"], help='Context to use for the text modality')
    parser.add_argument('--max_patience', type=int, default=50, help='Max patience for early stopping')
    parser.add_argument('--textualize_audio_levels', type=bool, default=True, help='Should the audio feature scores be textualized in the prompt')
    parser.add_argument('--skipped_frames', type=int, default=None, help='Number of frames to skip during action unit pre-processing')
    parser.add_argument('--used_folds', type=int, nargs='+', default=[0], help='Folds to use for training')
    parser.add_argument('--model', type=str, default='llama3', choices=['llama3', 'llama2'], help='Name of the model "llama3", "llama2".')
    parser.add_argument('--seed', type=int, default=0, help='Random seeds to use for training')
    parser.add_argument('--dataset', type=str, default='C-EXPR-DB', help='Dataset to use for training')
    parser.add_argument('--training_method', type=str, default='windows', choices=['windows', 'all'], help='Training method to use')
    parser.add_argument('--use_other_class', type=bool, default=False, help='Use the "Other" class in C-EXPR-DB training')
    parser.add_argument('--window_size', type=int, default=20, help='Window size for training with windows')
    parser.add_argument('--window_hop', type=int, default=10, help='Window hop size for training with windows')
    parser.add_argument('--use_meld_train_subset', type=bool, default=False, help='Use MELD train subset')
    parser.add_argument('--use_single_frame_per_video', type=bool, default=True, help='Use single frame per video for MELD')
    parser.add_argument('--used_modalities', type=str, nargs='+', default=['T', 'V', 'A'], help='List of Modalities to use, "T" for Textual, "V" for visual, "A" for Audio.')
    parser.add_argument('--hume_supp_context', type=int, default=2, help='supplementary context (in seconds) taken for clips in C-EXPR-DB')
    return parser.parse_args()

