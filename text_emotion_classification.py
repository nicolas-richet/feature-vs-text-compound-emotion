
from torch.utils.data import DataLoader, RandomSampler, WeightedRandomSampler
import constants
import torch
import torch.nn as nn
import numpy as np
import datetime
import random
import matplotlib.pyplot as plt
import pandas as pd
from frame_data_processing import preprocess_data
from model import get_model, train, get_predictions, reinitialize_weights
from sklearn.metrics import f1_score
from time import sleep
plt.style.use("ggplot")
from parser import parse_args

def main(args):
    torch.cuda.set_device(f'cuda:{args.gpu_id}')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device used: {device}')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    base_model_name = args.model
    # Process data
    split_dict_list = preprocess_data(device=device, args=args, tokenizer_name=base_model_name)
    
    batch_size = args.batch_size
    lr = args.lr
    for split_dict in split_dict_list:
        split_name = split_dict['split']
        time = datetime.datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
        model_name = f'{base_model_name}_{args.dataset}_{split_name}_{time}'
        args.model_name = model_name
        sampler = None
        train_dataloader = DataLoader(split_dict['train'], sampler = sampler, batch_size=batch_size, shuffle=True)
        #test_dataloader = DataLoader(split_dict['test'], batch_size=batch_size)
        dev_dataloader = DataLoader(split_dict['val'], batch_size=batch_size, shuffle=False)
        # Load Model
        model = get_model(base_model_name, args = args)
        if args.init_model:
            model.load_state_dict(torch.load(args.init_model_path))
            print(f'model loaded : {args.init_model_path}')
            print(model.base_model.model.score.weight[0, :10])
            sleep(10)
            # Access the classification head
            classification_head = model.base_model.model.score
            # Apply the reinitialization
            classification_head.apply(reinitialize_weights)
            print('Classification head reinitialized')
            print(model.base_model.model.score.weight[0, :10])
            sleep(10)
            print(model)
        sleep(10)
        model.to(device)
        
        log_file = f'{constants.OUTPUT_PATH}/{model_name}_log.txt'
        with open(log_file, 'w')as f:
            f.write(f'model: {model_name}\ntraining_arguments: {args}')
        # Train Model
        training_stats = train(model = model, train_dataloader=train_dataloader, dev_dataloader=dev_dataloader, device=device, 
                            model_name=model_name, args = args, log_file = log_file)
        
        # Plot Results
        training_loss = [x['Training Loss'] for x in training_stats]
        valid_loss = [x['Valid. Loss'] for x in training_stats]
        if args.dataset == 'C-EXPR-DB':
            valid_f1_wo_other = [x['Valid. F1 wo. Other'] for x in training_stats]
            train_f1 = [x['Train F1'] for x in training_stats]
            fig = plt.figure(figsize = (12,8))
            f1_ax = fig.add_subplot(2,1,1)
            f1_ax.plot(range(len(valid_f1_wo_other)), valid_f1_wo_other, label = 'Validation F1 wo. Other')
            f1_ax.plot(range(len(train_f1)), train_f1, label = 'Train F1')
            f1_ax.set_xlabel(f'Epochs')

        if args.dataset == 'MELD':
            valid_f1 = [x['Valid. F1'] for x in training_stats]
            valid_acc = [x['Valid. Acc'] for x in training_stats]
            fig = plt.figure(figsize = (12,8))
            f1_ax = fig.add_subplot(2,1,1)
            f1_ax.plot(range(len(valid_f1)), valid_f1, label ='Validation F1')
            f1_ax.plot(range(len(valid_acc)), valid_acc, label = 'Validation Accuracy')
            f1_ax.set_xlabel(f'Epochs')

        
        f1_ax.set_title(f'Model scoring evolution {split_name}\n BZ:{batch_size} LR:{lr} MODEL:{model_name} SEED:{args.seed}')
        f1_ax.legend()
        acc_ax = fig.add_subplot(2,1,2)
        acc_ax.plot(range(len(training_loss)), training_loss, label ="Training Loss")
        acc_ax.plot(range(len(valid_loss)), valid_loss, label = 'Validation Loss')
        acc_ax.set_title(f'Loss Evolution')
        acc_ax.legend()
        plt.savefig(f'{constants.OUTPUT_PATH}/LossEvolution{time}')
        plt.close()                    
        training_stats_df = pd.DataFrame.from_records(training_stats)
        training_stats_df.to_pickle(f'{constants.OUTPUT_PATH}/stats_{time}.pkl')
        with open(f'{constants.OUTPUT_PATH}/results.txt', 'a') as result_file:
            if args.dataset == 'C-EXPR-DB':
                log = f'\nbest val f1 wo. Other {np.max(valid_f1_wo_other)} split:{split_name}; '
                #log = f'\nbest val f1 wo. Other {np.max(valid_f1_wo_other)} split:{split_name} window_size:{args.window_size} MODALITIES:{args.used_modalities} BZ:{args.batch_size} LR:{args.lr} TRANSCRIPTION_CONTEXT: {constants.TRANSCRIPTION_CONTEXT} MODEL:{model_name} SEED:{seed} wd:{constants.PARAM_GRID["weight_decay"]} {time}  lora_r:{constants.LORA_R} lora_alpha:{constants.LORA_ALPHA} lora_dropout:{constants.LORA_DROPOUT} max grad norm:{constants.MAX_GRAD_NORM} samples_per_epochs:{constants.SAMPLES_PER_EPOCHS}'
                #log += f' Other class:{constants.USE_OTHER_CLASS}'
            elif args.dataset == 'MELD':
                log = f'\nbest val f1 {np.max(valid_f1)}'
            log += f'{args}'
            result_file.write(log) 
        model = ''
        model = get_model(model_name=base_model_name, args=args)
        model.load_state_dict(torch.load('/home/ens/AU58490/work/YP/text-emotion/models/' +model_name))
        model.to(device)
        model.eval()
        predictions = get_predictions(model, dev_dataloader, device)
        print('f1:', f1_score(split_dict['val'][:][2], predictions, average = 'weighted'))  

if __name__ == "__main__":
    args = parse_args()
    print(args)
    sleep(5)
    main(args)
