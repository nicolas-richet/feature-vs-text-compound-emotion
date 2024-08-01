
from torch.utils.data import DataLoader, RandomSampler, WeightedRandomSampler
import constants
import torch
import numpy as np
import datetime
import random
import matplotlib.pyplot as plt
import pandas as pd
from frame_data_processing import preprocess_data
from model import get_model, train
from time import sleep
plt.style.use("ggplot")

def main():
    torch.cuda.set_device(f'cuda:{constants.GPU_ID}')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device used: {device}')
    for seed in constants.SEEDS:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        for model_name in constants.MODELS:
            base_model_name = model_name
            # Process data
            split_dict_list = preprocess_data(device=device, tokenizer_name=base_model_name)
            
            for batch_size in constants.PARAM_GRID['batch_size']:
                for lr in constants.PARAM_GRID['lr']:
                    print(f"MODEL: {model_name}")
                    print(f"BATCH_SIZE: {batch_size}")
                    print(f"LR: {lr}")
                    for split_dict in split_dict_list:
                        split_name = split_dict['split']
                        time = datetime.datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
                        model_name = f'{base_model_name}_{constants.DATASET}_{split_name}_{time}'
                        labels = split_dict['train'][:][2]
                        #class_weights = [1/len(np.where(labels ==k)[0]) for k in np.unique(labels)]
                        #samples_weights = np.array([class_weights[k] for k in labels])
                        #samples_weights = torch.from_numpy(samples_weights)
                        #sampler = WeightedRandomSampler(samples_weights, num_samples=constants.SAMPLES_PER_EPOCHS)
                        sampler = None
                        train_dataloader = DataLoader(split_dict['train'], sampler = sampler, batch_size=batch_size, shuffle=True)
                        #test_dataloader = DataLoader(split_dict['test'], batch_size=batch_size)
                        dev_dataloader = DataLoader(split_dict['val'], batch_size=batch_size)
                        # Load Model
                        model = get_model(base_model_name)
                        sleep(10)
                        model.to(device)
                        
                        # Train Model
                        training_stats = train(model, train_dataloader, dev_dataloader, device, lr =lr, 
                                            weight_decay=constants.PARAM_GRID['weight_decay'], 
                                            model_name=model_name, max_patience = constants.MAX_PATIENCE)
                        
                        # Plot Results
                        training_loss = [x['Training Loss'] for x in training_stats]
                        valid_loss = [x['Valid. Loss'] for x in training_stats]
                        if constants.DATASET == 'C-EXPR-DB':
                            valid_f1_wo_other = [x['Valid. F1 wo. Other'] for x in training_stats]
                            train_f1 = [x['Train F1'] for x in training_stats]
                            fig = plt.figure(figsize = (12,8))
                            f1_ax = fig.add_subplot(2,1,1)
                            f1_ax.plot(range(len(valid_f1_wo_other)), valid_f1_wo_other, label = 'Validation F1 wo. Other')
                            f1_ax.plot(range(len(train_f1)), train_f1, label = 'Train F1')
                            f1_ax.set_xlabel(f'Epochs')

                        if constants.DATASET == 'MELD':
                            valid_f1 = [x['Valid. F1'] for x in training_stats]
                            valid_acc = [x['Valid. Acc'] for x in training_stats]
                            fig = plt.figure(figsize = (12,8))
                            f1_ax = fig.add_subplot(2,1,1)
                            f1_ax.plot(range(len(valid_f1)), valid_f1, label ='Validation F1')
                            f1_ax.plot(range(len(valid_acc)), valid_acc, label = 'Validation Accuracy')
                            f1_ax.set_xlabel(f'Epochs')

                        
                        f1_ax.set_title(f'Model scoring evolution {split_name}\n BZ:{batch_size} LR:{lr} MODEL:{model_name} SEED:{seed}')
                        f1_ax.legend()
                        acc_ax = fig.add_subplot(2,1,2)
                        acc_ax.plot(range(len(training_loss)), training_loss, label ="Training Loss")
                        acc_ax.plot(range(len(valid_loss)), valid_loss, label = 'Validation Loss')
                        acc_ax.set_title(f'Loss Evolution')
                        acc_ax.legend()
                        plt.savefig(f'{constants.UNIMODAL_TEXT_OUTPUT_PATH}/LossEvolution{time}')
                        plt.close()                    
                        training_stats_df = pd.DataFrame.from_records(training_stats)
                        training_stats_df.to_pickle(f'{constants.UNIMODAL_TEXT_OUTPUT_PATH}/stats_{time}.pkl')
                        with open(f'{constants.UNIMODAL_TEXT_OUTPUT_PATH}/results.txt', 'a') as result_file:
                            if constants.DATASET == 'C-EXPR-DB':
                                log = f'\nbest val f1 wo. Other {np.max(valid_f1_wo_other)} split:{split_name} window_size:{constants.WINDOW_SIZE} MODALITIES:{constants.USED_MODALITIES} BZ:{batch_size} LR:{lr} TRANSCRIPTION_CONTEXT: {constants.TRANSCRIPTION_CONTEXT} MODEL:{model_name} SEED:{seed} wd:{constants.PARAM_GRID["weight_decay"]} {time}  lora_r:{constants.LORA_R} lora_alpha:{constants.LORA_ALPHA} lora_dropout:{constants.LORA_DROPOUT} max grad norm:{constants.MAX_GRAD_NORM} samples_per_epochs:{constants.SAMPLES_PER_EPOCHS}'
                                log += f' Other class:{constants.USE_OTHER_CLASS}'
                            elif constants.DATASET == 'MELD':
                                log = f'\nbest val f1 {np.max(valid_f1)} 100%/single_frame {constants.USE_SINGLE_FRAME_PER_VIDEO} split:{split_name} window_size:{constants.WINDOW_SIZE} hop_size:{constants.WINDOW_HOP} MODALITIES:{constants.USED_MODALITIES} BZ:{batch_size} LR:{lr}  TRANSCRIPTION_CONTEXT: {constants.TRANSCRIPTION_CONTEXT} MODEL:{model_name} SEED:{seed} wd:{constants.PARAM_GRID["weight_decay"]} {time}  lora_r:{constants.LORA_R} lora_alpha:{constants.LORA_ALPHA} lora_dropout:{constants.LORA_DROPOUT} max grad norm:{constants.MAX_GRAD_NORM} samples_per_epochs:{constants.SAMPLES_PER_EPOCHS}'
                            result_file.write(log)    
main()
