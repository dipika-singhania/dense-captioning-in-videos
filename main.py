from __future__ import print_function, division
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F
import argparse
from torch.utils.data.dataloader import default_collate
import json
import glob
from my_models import network
import torchtext
from my_dataloaders import activity_net
import time
from torchtext.data.metrics import bleu_score
import arguments

args = arguments.get_args()

####################### Need to add them to to argumnets ############
epochs = args.epoch
batch_size = args.batch_size
workers = args.num_workers
learning_rate = args.lr
lr_end = args.lr_end
feature_dim = args.feature_dim
linear_dim = args.linear_dim
embed_dim = args.embed_dim

dataset_file = args.dataset_file
features_base_dir = args.features_base_dir
duration_file = args.duration_file
model_file_path = args.model_file_path

threshold = args.threshold  # Use this threshold for the inference time

iou_threshold = args.threshold  # Used only for training
teacher_forcing_ratio = args.iou_threshold

non_local_params = {'scale_factor': args.scale_factor, 'scale': args.scale,
                    'latent_dim': args.latent_dim, 'dropout_rate': args.dropout_rate}

if args.dim_past_list is None or len(args.dim_past_list) == 0:
    args.dim_past_list = [10, 15, 20]

if args.dim_cur_list is None or len(args.dim_past_list) == 0:
    args.dim_cur_list = [30, 50]

event_proposal_model_param_dict = {'dim_past_list': args.dim_past_list, 'dim_cur_list': args.dim_cur_list, \
                                   'max_proposals': args.max_proposals, 'threshold': args.threshold}
encoding_proposal_model_param_dict = {'max_proposals': args.max_proposals, 'recent_list': [0, 1, 2], \
                                      'dim_curr': [2], 'threshold': args.threshold}
captioning_proposal_model_param_dict = {'max_proposals': args.max_proposals, 'max_sentence_len': args.max_sentence_len, \
                                        'teacher_forcing_ratio': args.teacher_forcing_ratio}

dataset = args.dataset
resume = args.resume

feature_names = ['_bn']

################ Some global parameters defined ###############################
g_start_epoch = 0
g_best_perf = 0
clip = 1
lr_log_ratio = np.log(lr_end / learning_rate)
epsilon = 0.00001
######################## Building Vocab and building some local variables ###########
data_vocab, json_data, _ = activity_net.get_vocab_and_sentences(dataset_file)

vocab_size = len(data_vocab.vocab)
pad_idx = data_vocab.vocab.stoi['<pad>']

print("\nEvent Proposal model parameters dictionary\n", event_proposal_model_param_dict)
print("\nEncoding Proposal model parameters dictionary\n", encoding_proposal_model_param_dict)
print("\nCaptioning Proposal model parameters dictionary\n", captioning_proposal_model_param_dict)

########################################################################################
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def learning_rate_mod_factor(p_epoch):
    t = p_epoch / (epochs * 1.0)
    if t < 0.1:
        factor = .0
    elif t <= 0.9:
        factor = lr_log_ratio * (t - 0.1) / 0.8
    else:
        factor = lr_log_ratio
    return np.exp(factor)


def criterion(pred, target):
    pos_loss = 8.00 * target * torch.log(pred + epsilon)
    neg_loss = (1.0 - target) * torch.log(1 - pred + epsilon) 
    value = (-1.00) * (torch.mean(pos_loss) + torch.mean(neg_loss))
    return value


def load_checkpoint(p_model, p_name, best=False):
    if best:
        chk = torch.load(os.path.join(model_file_path, p_name + '_best.pth.tar'))
    else:
        chk = torch.load(os.path.join(model_file_path, p_name + '.pth.tar'))

    epoch = chk['epoch']
    best_perf = chk['best_perf']
    perf = chk['perf']

    p_model.load_state_dict(chk['state_dict'])

    return epoch, perf, best_perf


def save_model(p_model, p_name, p_epoch, p_perf, p_best_perf, is_best=False):
    torch.save({'state_dict': p_model.state_dict(), 'epoch': p_epoch,
                'perf': p_perf, 'best_perf': p_best_perf}, os.path.join(model_file_path, p_name + '.pth.tar'))
    if is_best:
        torch.save({'state_dict': p_model.state_dict(), 'epoch': p_epoch, 'perf': p_perf, 'best_perf': p_best_perf}, os.path.join(
                     model_file_path, p_name + '_best.pth.tar'))


def make_model_name():
    if not os.path.exists(model_file_path):
        os.mkdir(model_file_path)
    name = 'adam_cap_{}'.format(dataset)
    name += '_ft{}'.format('-'.join(feature_names))
    name += '_ep{}_bs{}_lr{}_th_{}'.format(epochs, batch_size, learning_rate, threshold)
    name += '_fd{}_ld{}_nlld{}_nldrp{}'.format(feature_dim, linear_dim, non_local_params['latent_dim'], non_local_params['dropout_rate'])
    name += '_en_dim_past' + "-".join(map(str, event_proposal_model_param_dict['dim_past_list']))
    name += '_en_dim_curr' + "-".join(map(str, event_proposal_model_param_dict['dim_cur_list']))
    name += '_en_maxp{}'.format(args.max_proposals)
    return name


def cal_acc(probs, target, threshold=encoding_proposal_model_param_dict['threshold']):
    acc = 0
    for i, ele in enumerate(target):
        predicted = (probs[i] > threshold) 
        intersection = predicted * ele.type(torch.ByteTensor).to(device)
        union_target = (ele == 1) | (predicted == 1)
        acc += torch.sum(intersection) / torch.sum(union_target.type(torch.FloatTensor).to(device))
    acc = acc / len(target) * 100
    return acc


def train_one_pass(p_network, p_loader, p_optim, criterion_sentence):
    # Input: Network and Optimizer
    # Output: Averge accuracy , Avergae loss in the pass
    p_network.train()
    acc_one_pass = []
    loss_one_pass = []
    for i, ele in enumerate(iter(p_loader)):
        recent_features = []
        for ele_r in ele['recent_features']:
            recent_features.append(ele_r.type(torch.FloatTensor).to(device))
        
        past_features = []
        for ele_p in ele['past_features']:
            past_features.append(ele_p.type(torch.FloatTensor).to(device))
        
        labels = []
        for ele_l in ele['label']:
            labels.append(ele_l.type(torch.FloatTensor).to(device))
        
        sentences = ele['sentences'].type(torch.LongTensor).to(device)
        
        # input_sentences, x_past_actual_all, x_curr_actual_all, labels, is_train, teacher_forcing_ratio
        pred_list, split_size_or_sections, outputs, trg, filter_used = p_network(sentences, past_features, recent_features, 
                                                                                       labels, True, 
                                                                                       teacher_forcing_ratio)  

        loss = torch.tensor(0, dtype=torch.float32).to(device)
        local_loss_list = []
        for index, pred in enumerate(pred_list):
            temp_loss = criterion(pred, labels[index])
            local_loss_list.append(temp_loss.item())
            loss = loss + temp_loss


        vocab_loss = torch.tensor(0, dtype=torch.float32).to(device)
        if outputs is not None:
            output_dim = outputs.shape[-1]
            outputs_flt = outputs[:, 1:].contiguous().view(-1, 1, output_dim).squeeze()
            trg_flt = trg[:, 1:].contiguous().view(-1, 1).squeeze()

            vocab_loss = criterion_sentence(outputs_flt,trg_flt) 

        total_loss = loss + vocab_loss
        p_optim.zero_grad()
        total_loss.backward()

        torch.nn.utils.clip_grad_norm_(p_network.parameters(), clip)

        p_optim.step()
        with torch.no_grad():  
            acc = cal_acc(pred_list, labels)
            if i % 100 == 0:
                print("Iter: {:2.2f}. ".format(i / len(p_loader)))
                for li, lt in enumerate(local_loss_list):
                    print("Localization loss {:3d}: {:3.2f}. ".format(li, lt))
                print("Vocab loss: {:3.2f}. ".format(vocab_loss.item()),
                      "localization loss: {:3.2}. ".format(loss.item()),
                      "Train Total Loss: {:3.2f}. ".format(total_loss.item()),
                      "Train Accuracy: {:3.2f}% ".format(acc)
                     )
                if trg is not None and outputs is not None:
                    elements_non_zero = split_size_or_sections[split_size_or_sections > 0]
                    elements_non_zero = [int(ele) for ele in elements_non_zero.tolist()]
                    splitted_sentences_gt = torch.split(trg, elements_non_zero)
                    splitted_sentences_pred = torch.split(outputs, elements_non_zero)
                    for s_i, sentence in enumerate(splitted_sentences_gt[0]):
                        trg_sentence = [data_vocab.vocab.itos[i] for i in sentence] 
                        print("Target sentence: ", s_i, "is ", " ".join(trg_sentence))
                    for s_i, sentence_prob in enumerate(splitted_sentences_pred[0]):
                        sentence = sentence_prob.argmax(1)
                        trg_sentence = [data_vocab.vocab.itos[i] for i in sentence] 
                        print("Source sentence: ", s_i, "is ", " ".join(trg_sentence))
            acc_one_pass.append(acc)
            loss_one_pass.append(total_loss.item())
    return sum(acc_one_pass) / len(acc_one_pass), sum(loss_one_pass) / len(loss_one_pass)


def val_one_pass(p_network, p_loader, criterion_sentence):
    # Input: Network and Optimizer
    # Output: Averge accuracy , Avergae loss in the pass
    p_network.eval()
    acc_one_pass = []
    loss_one_pass = []
    for i, ele in enumerate(iter(p_loader)):
        recent_features = []
        for ele_r in ele['recent_features']:
            recent_features.append(ele_r.type(torch.FloatTensor).to(device))
        
        past_features = []
        for ele_p in ele['past_features']:
            past_features.append(ele_p.type(torch.FloatTensor).to(device))
        
        labels = []
        for ele_l in ele['label']:
            labels.append(ele_l.type(torch.FloatTensor).to(device))

        sentences = ele['sentences'].type(torch.LongTensor).to(device)
        
        with torch.no_grad():
            # input_sentences, x_past_actual_all, x_curr_actual_all, labels, is_train, teacher_forcing_ratio
            pred_list, split_size_or_sections, outputs, trg, filter_used = p_network(None, past_features, recent_features, 
                                                                                           labels, False, 
                                                                                           teacher_forcing_ratio)  
            acc = cal_acc(pred_list, labels)
            loss = torch.tensor(0, dtype=torch.float32).to(device)
            local_loss_list = []
            for index, pred in enumerate(pred_list):
                temp_loss = criterion(pred, labels[index])
                local_loss_list.append(temp_loss.item())
                loss = loss + temp_loss


            vocab_loss = torch.tensor(0, dtype=torch.float32).to(device)
            if outputs is not None and filter_used is not None:
                trg = sentences.view(-1, 1, sentences.shape[2])
                trg = trg[filter_used].squeeze()
                output_dim = outputs.shape[-1]
                outputs_flt = outputs[:, 1:].contiguous().view(-1, 1, output_dim).squeeze()
                trg_flt = trg[:, 1:].contiguous().view(-1, 1).squeeze()

                vocab_loss = criterion_sentence(outputs_flt, trg_flt) 


            total_loss = loss + vocab_loss

            if i % 500 == 0:
                print("Iter: {:2.2f}. ".format(i / len(p_loader)))
                for li, lt in enumerate(local_loss_list):
                    print("Localization loss {:3d}: {:3.2f}. ".format(li, lt))
                print("Vocab loss: {:3.2f}. ".format(vocab_loss.item()),
                      "localization loss: {:3.2}. ".format(loss.item()),
                      "Total validation Loss: {:3.2f}. ".format(total_loss.item()),
                      "Validation Accuracy: {:3.2f}% ".format(acc)
                     )
                if filter_used is not None and outputs is not None:
                    trg = sentences.view(-1, 1, sentences.shape[2])
                    trg = trg[filter_used].squeeze()
                    elements_non_zero = split_size_or_sections[split_size_or_sections > 0]
                    elements_non_zero = [int(ele) for ele in elements_non_zero.tolist()]
                    splitted_sentences_gt = torch.split(trg, elements_non_zero)
                    splitted_sentences_pred = torch.split(outputs, elements_non_zero)
                    for s_i, sentence in enumerate(splitted_sentences_gt[0]):
                        trg_sentence = [data_vocab.vocab.itos[i] for i in sentence] 
                        print("Target sentence: ", s_i, "is ", " ".join(trg_sentence))
                    for s_i, sentence_prob in enumerate(splitted_sentences_pred[0]):
                        sentence = sentence_prob.argmax(1)
                        trg_sentence = [data_vocab.vocab.itos[i] for i in sentence] 
                        print("Source sentence: ", s_i, "is ", " ".join(trg_sentence))

            acc = cal_acc(pred_list, labels)
            acc_one_pass.append(acc)
            loss_one_pass.append(total_loss.item())
    return sum(acc_one_pass) / len(acc_one_pass), sum(loss_one_pass) / len(loss_one_pass)

def multiple_pass(p_network, p_optim, p_trainloader, p_valloader, p_name):
    # Input: p_network, p_optim
    # Output: Output a chart accuracy 
    global g_start_epoch, g_best_perf
    all_epochs_acc = []
    all_epochs_loss = []
    l_is_best = False
    pos_weight = torch.tensor([8, 2]).type(torch.FloatTensor).to(device)

    TRG_PAD_IDX = data_vocab.vocab.stoi[data_vocab.pad_token]
    criterion_sentence = nn.CrossEntropyLoss(ignore_index = TRG_PAD_IDX)

    for epoch in range(g_start_epoch, epochs):
        start_time = time.time()
        # factor = learning_rate_mod_factor(epoch)
        # for i, g in enumerate(p_optim.param_groups):
        #    g['lr'] = learning_rate * factor
        #    print("Learning rate for param %d is currently %f" %(i, g['lr']))

        avg_acc, avg_loss = train_one_pass(p_network, p_trainloader, p_optim, criterion_sentence)
        print('\033[92m',
              "Epoch: {:d}. ".format(epoch),
              "Training Total Loss: {:.2f}. ".format(avg_loss),
              "Training Accuracy: {:.2f}% ".format(avg_acc),
              '\033[0m')
      
        val_accuracy, val_loss = val_one_pass(p_network, p_valloader, criterion_sentence)
        if g_best_perf < val_accuracy:
            g_best_perf = val_accuracy
            l_is_best = True
        else:
            l_is_best = False

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        with open(model_file_path + '/'  +  p_name + '.txt', 'a') as f:
            f.write("%d - %0.2f\n" %  (epoch + 1, val_accuracy) )

        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print('\033[92m',
                "Epoch: {:d}. ".format(epoch),
                "Val Total Loss: {:.2f}. ".format(val_loss),
                "Val Accuracy: {:.2f}% ".format(val_accuracy),
                "Best Val Accuracy: {:.2f}% ".format(g_best_perf),
                '\033[0m')

        # save checkpoint at the end of each train/val epoch
        save_model(p_network, p_name, epoch + 1, val_accuracy, g_best_perf, is_best=l_is_best)

        all_epochs_loss.append(val_loss)
        all_epochs_acc.append(val_accuracy)

    with open(model_file_path + '/'  +    p_name + '.txt', 'a') as f:
        f.write("%d - %0.2f\n" %  (epoch + 1, g_best_perf))
    # Plotting the accuracy and loss graphs
    print("all_epochs_acc = \n", all_epochs_acc)
    print("all epochs loss = \n", all_epochs_loss) 


################Model Name Creation######################
g_name = make_model_name()

#############################Declaring dataset###################################### 

train_dataset = activity_net.ActivityNet(json_data, duration_file, os.path.join(features_base_dir + 'training'), feature_names,
                 event_proposal_model_param_dict, encoding_proposal_model_param_dict, captioning_proposal_model_param_dict, data_vocab)
val_dataset = activity_net.ActivityNet(json_data, duration_file, os.path.join(features_base_dir + 'validation'), feature_names,
                 event_proposal_model_param_dict, encoding_proposal_model_param_dict, captioning_proposal_model_param_dict, data_vocab)
trainloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=workers,
           pin_memory=True, drop_last=True, shuffle=True, collate_fn=activity_net.my_collate)                                                                 
valloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=workers,
           pin_memory=True, drop_last=True, shuffle=False, collate_fn=activity_net.my_collate)                                                                 


############################Declaring Models############################################
# encoding_network = network.EncodingProposal(non_local_params, event_proposal_model_param_dict, linear_dim, feature_dim)
encoding_network = network.OverallNetwork(non_local_params, event_proposal_model_param_dict, encoding_proposal_model_param_dict, \
                                          captioning_proposal_model_param_dict, feature_dim, linear_dim, embed_dim, vocab_size, \
                                          pad_idx, iou_threshold, data_vocab)
encoding_network.to(device)                                                                      

count_params = sum([p.numel() for p in encoding_network.parameters()])
print("Number of paramters in network ", count_params/1e6)
    
### loading the last checkpoint to resume training
if resume is True:
    g_start_epoch, curr_perf, g_best_perf = load_checkpoint(encoding_network, g_name)
    print("Loaded checkpoint with epoch {}, curr_perf {} and best perf {}".format(g_start_epoch, curr_perf, g_best_perf))

optimizer = torch.optim.Adam(encoding_network.parameters(), lr=learning_rate, betas=(0.99, 0.999))

# optimizer = torch.optim.SGD(encoding_network.parameters(), lr=learning_rate, momentum=0.9)
# avg_acc, avg_loss = train_one_pass(encoding_network, optimizer, trainloader)

##############Train and validate multple pass#####
multiple_pass(encoding_network, optimizer, trainloader, valloader, g_name)


###################Finding the learning rate###########
# from lr_finder import LRFinder
# lr_finder = LRFinder(model=encoding_network, optimizer=optimizer, criterion=criterion, device=device)
# lr_finder.range_test(trainloader, end_lr=10, num_iter=200, step_mode="exp")
# lr_finder.plot(fname='lr_probing.pdf')




