import constants
from transformers import BertForSequenceClassification, get_linear_schedule_with_warmup,  get_polynomial_decay_schedule_with_warmup, RobertaForSequenceClassification, AutoModelForSequenceClassification, BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer
import bitsandbytes as bnb
# Phi3ForSequenceClassification,
import time
import torch
from torch import nn
import numpy as np
import datetime
from sklearn.metrics import f1_score, confusion_matrix
from peft import LoraConfig, TaskType, get_peft_model
from torchinfo import summary
import sys, os
from my_lr_scheduler import MyStepLR
import yaml
from pathlib import Path

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

def get_model(model_name, args, verbose = True):
    """
    Returns the pretrained model to be fine-tuned
    Args:
        model_name (str): name of the model
            Available Models: 'bert', 'roberta', 'llama2', 'llama3', 'phi2', 'phi-3-zeroshot', 'stablelm-zeroshot'.
        verbose (boolean): prints a summary of the model.
    """
    if args.dataset == 'MELD':
        labels = constants.EMOTIONS
    elif args.dataset == 'C-EXPR-DB':
        if args.use_other_class:
            labels = constants.COMPOUND_EMOTIONS
        else:
            labels = yaml.safe_load(Path('/datasets/C-EXPR-DB/folds/split-0/class_id.yaml').read_text())
    quantization_config = BitsAndBytesConfig(
        load_in_4bit = True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype = torch.float16,
        bnb_4bit_quant_type="nf4"
    )
    
    if model_name == 'phi-3-zeroshot':
        model = AutoModelForCausalLM.from_pretrained(
            "microsoft/Phi-3-mini-4k-instruct",
            trust_remote_code = True,
            quantization_config=quantization_config,
            device_map="auto")

    elif model_name == 'stablelm-zeroshot':
        model = AutoModelForCausalLM.from_pretrained(
            'stabilityai/stablelm-2-1_6b-chat',
            quantization_config=quantization_config,
            device_map="auto"
        )

    elif model_name == 'bert':
        model = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased", 
            num_labels = len(labels), 
            output_attentions =False, 
            output_hidden_states = False)
    elif model_name == 'roberta':
        #model = RobertaForSequenceClassification.from_pretrained(
        #    "cardiffnlp/twitter-roberta-base-emotion", 
        #    num_labels = len(constants.EMOTIONS), 
        #    ignore_mismatched_sizes=True)
        model = RobertaForSequenceClassification.from_pretrained(
            "cardiffnlp/twitter-roberta-large-emotion-latest", 
            num_labels = len(labels), 
            ignore_mismatched_sizes=True, problem_type = 'single_label_classification')
    elif model_name == 'llama2':
        model = AutoModelForSequenceClassification.from_pretrained(
            "meta-llama/Llama-2-7b-hf", 
            num_labels = len(labels),
            token = constants.LLAMA_TOKEN, 
            quantization_config=quantization_config, 
            torch_dtype = torch.bfloat16)
        model.config.pad_token_id = model.config.eos_token_id

    elif model_name == 'llama3':
        model = AutoModelForSequenceClassification.from_pretrained(
            "meta-llama/Meta-Llama-3-8B", 
            num_labels = len(labels),
            token = constants.LLAMA_TOKEN, 
            quantization_config=quantization_config, 
            torch_dtype = torch.bfloat16)
        model.config.pad_token_id = model.config.eos_token_id

    elif model_name == 'llama2-zeroshot':
        model = AutoModelForCausalLM.from_pretrained(
            'meta-llama/Llama-2-7b-hf',
            quantization_config=quantization_config,
            device_map="auto"
        )
    
    elif model_name == 'phi-2':
        model = AutoModelForSequenceClassification.from_pretrained("microsoft/phi-2", num_labels = len(labels), quantization_config=quantization_config, torch_dtype = torch.bfloat16, trust_remote_code = True)
        model.config.pad_token_id = model.config.eos_token_id

    else:
        model = None
    
    if model_name in ['llama2', 'phi-2', 'llama3', 'phi-3', 'roberta']:
        peft_config = LoraConfig(task_type=TaskType.SEQ_CLS, r = args.lora_r, lora_alpha = args.lora_alpha, lora_dropout =args.lora_dropout, use_rslora = True)
        model = get_peft_model(model, peft_config)
        if verbose:
            print(summary(model))
    return model



def score(labels, logits, metric = 'accuracy', average='weighted'):
    """
    Score the predictions with the given metric
    Args:
        labels (torch.Tensor): tensor of labels of shape (batch_size,)
        logits (torch.Tensor): tensor of logits of shape (batch_size, n) with n the number of labels
        metric (str): metric to use. Available : 'accuracy', 'f1'
        average (str): 'average' argument for f1-score
    """
    if metric == 'f1':
        print(confusion_matrix(labels, np.argmax(logits, axis=1)))
        return f1_score(labels, np.argmax(logits, axis=1),average=average)
    else:
        return (labels.flatten() == np.argmax(logits, axis=1)).sum()/labels.shape[0]

def format_time(elapsed):
    """
    Takes a time in seconds and returns a str hh:mm:ss
    Args:
        elapsed (float): time elapsed in seconds
    """
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

def get_predictions(model, dataloader, device):
    """
    Returns prediction of a given model using a dataloader
    Args:
        model (nn.Module): model to get predictions from
        dataloader (torch.DataLoader): DataLoader to use
        device (torch.device): device to use
    """
    logits_list = []
    for i, batch in enumerate(dataloader):
            print(f'batch {i}/{len(dataloader)}')
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            with torch.no_grad():        
                output= model(b_input_ids,  
                                    attention_mask=b_input_mask,
                                    labels=b_labels)
            logits = output.logits
            logits = logits.detach().cpu().to(torch.float16)
            logits_list.append(logits)
    logits = torch.cat(logits_list, 0)
    return np.argmax(logits, axis=1).numpy()


def reinitialize_weights(module):
    # Reinitialize the weights
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)

def train(model, train_dataloader, dev_dataloader, device, model_name, args, log_file):
    """
    Train the given model using the given train and dev dataloaders and device.
    Args:
        model (nn.Module): model to get predictions from.
        train_dataloader (torch.DataLoader): DataLoader to use to train.
        dev_dataloader (torch.DataLoader): DataLoader to use to validate.
        device (torch.device): device to use.
        model_name (str): name of the model used.
        args: training arguments
        log_file (str): path to the log_file
    """
    lr = args.lr
    weight_decay = args.weight_decay
    max_patience = args.max_patience

    patience = max_patience
    #optimizer = torch.optim.AdamW(model.parameters(), lr = lr, weight_decay=weight_decay)
    #optimizer = bnb.optim.PagedAdamW8bit(model.parameters(), lr = lr, weight_decay = weight_decay)
    optimizer = torch.optim.SGD(model.parameters(), lr = lr, weight_decay= weight_decay)
    epochs = args.epochs

    total_steps = len(train_dataloader) * epochs
    #scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps= total_steps)
    scheduler = MyStepLR(optimizer, step_size = 18, min_lr=1e-7, gamma=0.5)

    training_stats = []

    total_t0= time.time()
    best_val_f1_MELD =0
    best_val_f1_wo_other = 0
    for epoch in range(epochs):
        log = ''
        if patience == 0:
            print(f'Patience exceeded, Training stopped before epoch {epoch}')
            log +=f'\nPatience exceeded, Training stopped before epoch {epoch}'
            break
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch + 1, epochs))
        log +='\n======== Epoch {:} / {:} ========'.format(epoch + 1, epochs)
        print('Training...')
        log+='\nTraining...'
        t0 = time.time()
        total_train_loss = 0
        train_predictions_logits = []
        train_labels = []
        model.train()
        for batch in train_dataloader:
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            optimizer.zero_grad()
            output = model(b_input_ids, 
                                attention_mask=b_input_mask, 
                                labels=b_labels)
            loss = output.loss
            total_train_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.3)
            optimizer.step()
            logits = output.logits
            train_logits = logits.detach().cpu().to(torch.float16)
            label_ids = b_labels.to('cpu').numpy()
            train_labels.append(torch.tensor(label_ids))
            train_predictions_logits.append(train_logits)
        train_predictions_logits = torch.cat(train_predictions_logits, 0)
        train_labels = torch.cat(train_labels, 0)
        avg_train_loss = total_train_loss / len(train_dataloader)

        train_f1 = score(train_labels, train_predictions_logits, metric='f1', average='weighted')
        
        print('train w-F1 :', train_f1)    
        log += f'\ntrain w-F1 : {train_f1}'
        training_time = format_time(time.time() - t0)
        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        log+="\n  Average training loss: {0:.2f}".format(avg_train_loss)
        print("  Training epoch took: {:}".format(training_time))
        log+= "\n  Training epoch took: {:}".format(training_time)

        #Validation
        print("")
        print("Running Validation...")
        log += '\nRunning Validation...'

        t0 = time.time()
        model.eval()

        total_eval_loss = 0
        labels = []
        predictions_logits = []

        for batch in dev_dataloader:
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            with torch.no_grad():        
                output= model(b_input_ids,  
                                    attention_mask=b_input_mask,
                                    labels=b_labels)
            loss = output.loss
            total_eval_loss += loss.item()
            logits = output.logits
            logits = logits.detach().cpu().to(torch.float16)
            label_ids = b_labels.to('cpu').numpy()
            labels.append(torch.tensor(label_ids))
            predictions_logits.append(logits)
        labels = torch.cat(labels)
        predictions_logits = torch.cat(predictions_logits, 0)

        avg_val_loss = total_eval_loss / len(dev_dataloader)

        validation_time = format_time(time.time() - t0)

        if args.dataset == 'C-EXPR-DB':

            if args.use_other_class:
                predictions_logits_wo_other = predictions_logits[:,:-1]
            else:
                predictions_logits_wo_other = predictions_logits
            predictions_wo_other = np.argmax(predictions_logits_wo_other, axis=1)
            val_f1_wo_other = score(labels, predictions_logits_wo_other, metric='f1', average='weighted')
            val_acc_wo_other = score(labels, predictions_logits_wo_other, metric='accuracy')
            
            print('validation w-F1 wo. other:', val_f1_wo_other)
            log += f'\nvalidation w-F1 wo. other: {val_f1_wo_other}'
            print(f'best valid f1 : {best_val_f1_wo_other}')
            log += f'\nbest valid f1 : {best_val_f1_wo_other}'

            if val_f1_wo_other >= best_val_f1_wo_other:
                Path(f'{constants.MODEL_PATH}/saved').mkdir(parents=True, exist_ok=True)
                model.save_pretrained(f'{constants.MODEL_PATH}/saved/{model_name}', token=constants.LLAMA_TOKEN)
                torch.save(model.state_dict(), f'{constants.MODEL_PATH}/{model_name}')
                best_val_f1_wo_other = val_f1_wo_other
                patience =max_patience
                print(f'Patience = {patience}')
                log += f'\nPatience = {patience}'

            else:
                patience -=1
                print(f'Patience = {patience}')
                log += f'\nPatience = {patience}'

            training_stats.append(
            {
                'epoch': epoch + 1,
                'Training Loss': avg_train_loss,
                'Valid. Loss': avg_val_loss,
                'Valid. F1 wo. Other': val_f1_wo_other,
                'Valid. Acc wo. Other': val_acc_wo_other,
                'Train F1': train_f1,
                'Training Time': training_time,
                'Validation Time': validation_time,
                'Labels': labels,
                'Predictions wo. Other': predictions_wo_other
                
            }
        )
        elif args.dataset == 'MELD':
            predictions = np.argmax(predictions_logits, axis=1)

            val_f1 = score(labels, predictions_logits, metric='f1', average='weighted')
            val_acc = score(labels, predictions_logits, metric='accuracy')

            print('validation w-F1:', val_f1)
            log+=f'\nvalidation w-F1: {val_f1}'
            print(f'best valid f1 : {best_val_f1_MELD}')
            log += f'\nbest valid f1 : {best_val_f1_MELD}'
            if val_f1 >= best_val_f1_MELD:
                Path(f'{constants.MODEL_PATH}/saved').mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), f'{constants.MODEL_PATH}/{model_name}')
                model.save_pretrained(f'{constants.MODEL_PATH}/saved/{model_name}', token=constants.LLAMA_TOKEN)
                best_val_f1_MELD = val_f1
                patience =max_patience
                print(f'Patience = {patience}')
                log += f'\nPatience = {patience}'
            else:
                patience -=1
                print(f'Patience = {patience}')
                log += f'\nPatience = {patience}'
            training_stats.append(
            {
                'epoch': epoch + 1,
                'Training Loss': avg_train_loss,
                'Valid. Loss': avg_val_loss,
                'Valid. F1': val_f1,
                'Valid. Acc': val_acc,
                'Training Time': training_time,
                'Validation Time': validation_time,
                'Labels': labels,
                'Predictions': predictions
                
            }
        )
                
        print(f'current lr : {scheduler.get_lr()}')
        log += f'\ncurrent lr : {scheduler.get_lr()}'
        with open(log_file, 'a') as f:
            f.write(log)
        scheduler.step()

    print("")
    print("Training complete!")
    with open(log_file, 'a') as f:
            f.write(f'\nTraining complete!')
            f.write("\nTotal training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0))) 
    print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))
    return training_stats

def get_best_model(model_name):
    """
    Returns the best trained model
    Args:
        model_name (str): name of the model.
    """
    model = get_model(model_name, verbose =False)
    model.load_state_dict(torch.load(f'{constants.MODEL_PATH}/{model_name}_model'))
    print('Fine Tuned Model Loaded.\n')
    return model


