from __future__ import print_function, division
import os
import sys
import torch
import h5py
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
# from my_models import no_lstm_network
import torchtext
from my_dataloaders import activity_net
import time
from torchtext.data.metrics import bleu_score
from tensorboardX import SummaryWriter
import random
from utils import calculate_bleu_meteor_scores
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.bleu.bleu import Bleu
import arguments
from json import encoder
import sys

encoder.FLOAT_REPR = lambda o: format(o, '.2f')
########################## Basic Initialization  #########################
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
torch.manual_seed(1234)
torch.cuda.manual_seed(1234)
random.seed(1234)
np.random.seed(1234)
####################### Need to add them to to argumnets ############
args = arguments.get_args()

epochs = args.epochs
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

iou_threshold = np.array(args.iou_threshold)  # Used only for training
teacher_forcing_ratio = args.teacher_forcing_ratio

non_local_params = {'scale_factor': args.scale_factor, 'scale': args.scale,
                    'latent_dim': args.latent_dim, 'dropout_rate': args.dropout_rate}

if args.dim_past_list is None or len(args.dim_past_list) == 0:
    args.dim_past_list = [10, 15, 20]

if args.dim_cur_list is None or len(args.dim_past_list) == 0:
    args.dim_cur_list = [30, 50]

event_proposal_model_param_dict = {'dim_past_list': args.dim_past_list, 'dim_cur_list': args.dim_cur_list,
                                   'max_proposals': args.max_proposals, 'threshold': args.threshold}
encoding_proposal_model_param_dict = {'max_proposals': args.max_proposals, 'recent_list': [0, 1, 2],
                                      'dim_curr': [2], 'threshold': args.threshold}
captioning_proposal_model_param_dict = {'max_proposals': args.max_proposals, 'max_sentence_len': args.max_sentence_len,
                                        'teacher_forcing_ratio': args.teacher_forcing_ratio}

dataset = args.dataset
resume = args.resume
sentence_train_start = args.sentence_train_start
clip = 1
data_vocab, json_data, max_proposals = activity_net.get_vocab_and_sentences(dataset_file)
is_tensorboard = args.tensorboard
is_calculate_scores = args.cal_scores
print_results = True
pick_best = False

def remove_nonascii(text):
    return ''.join([i if ord(i) < 128 else ' ' for i in text])

# if args.log_E is not None and len(args.log_E) != 0:
#     sys.stdout = open(args.log_E, 'a+')

vocab_size = len(data_vocab.vocab)
pad_idx = data_vocab.vocab.stoi['<pad>']
feature_names = args.feature_name
# feature_names = ['c3d']
################ Some global parameters defined ###############################

g_start_epoch = 0
g_best_perf = 0

lr_log_ratio = np.log(lr_end / learning_rate)

is_run_train_valid = True if args.mode == 'train' else False
is_run_valid_only = True if args.mode == 'validate' else False
print("\nEvent Proposal model parameters dictionary\n", event_proposal_model_param_dict)
print("\nEncoding Proposal model parameters dictionary\n", encoding_proposal_model_param_dict)
print("\nCaptioning Proposal model parameters dictionary\n", captioning_proposal_model_param_dict)
epsilon = 0.00001

######################################################################


def proposal_criterion(filtered_p, filtered_t):
    # filtered_t = target[target.sum(dim=2) > 0]
    # filtered_p = pred[target.sum(dim=2) > 0]
    pos_loss = filtered_t * torch.log(filtered_p + epsilon)
    neg_loss = (1.00 - filtered_t) * torch.log(1 - filtered_p + epsilon)
    value = (-1.00) * (torch.mean(pos_loss) + torch.mean(neg_loss))
    return value


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def load_checkpoint(p_model, p_name, p_optim, best=False):
    if best:
        file_name = os.path.join(model_file_path, p_name + '_best.pth.tar')
        chk = torch.load(file_name)
        print("Picking up best model with name = ", file_name)
    else:
        file_name = os.path.join(model_file_path, p_name + '.pth.tar')
        chk = torch.load(file_name)
        print("Picking up model with name = ", file_name)

    epoch = chk['epoch']
    best_perf = chk['best_perf']
    perf = chk['perf']

    print("Epoch picked up from = ", epoch, "best performance of model = {:2.2f}".format(best_perf),
          "last performance = {:2.2f}".format(perf))

    p_model.load_state_dict(chk['state_dict'])
    if 'optim_state_dict' in chk:
        p_optim.load_state_dict(chk['optim_state_dict'])

    return epoch, perf, best_perf


def save_model(p_model, p_name, p_epoch, p_perf, p_best_perf, p_optim, is_best=False):
    torch.save({'state_dict': p_model.state_dict(), 'epoch': p_epoch, 'optim_state_dict': p_optim.state_dict(),
                'perf': p_perf, 'best_perf': p_best_perf}, os.path.join(model_file_path, p_name + '.pth.tar'))
    if is_best:
        torch.save(
            {'state_dict': p_model.state_dict(), 'epoch': p_epoch, 'perf': p_perf, 'best_perf': p_best_perf,
             'optim_state_dict': p_optim.state_dict()}, os.path.join(model_file_path, p_name + '_best.pth.tar'))


def make_model_name():
    if not os.path.exists(model_file_path):
        os.mkdir(model_file_path)
    if args.no_tab_no_lstm:
        name = "no_tab_no_lstm_diff_sent_cap_{}".format(dataset)
    elif args.no_tab:
        name = "no_tab_sent_cap_{}".format(dataset)
    elif args.add_logit:
        # name = 'debug_remove_attn_logit_sent_cap_{}'.format(dataset)
        name = 'sent_cap_{}'.format(dataset)
        # name = 'Without_LSTM_logit_sent_cap_{}'.format(dataset)
    else:
        name = 'sent_regression_cap_{}'.format(dataset)
    name += '_ft{}'.format('-'.join(feature_names))
    name += '_ep{}_bs{}_lr{:.6f}_th_{:0.1f}'.format(epochs, batch_size, learning_rate, args.sentence_train_start)
    name += '_fd{}_ld{}_nlld{}_nldrp{:0.1f}'.format(feature_dim, linear_dim, non_local_params['latent_dim'],
                                               non_local_params['dropout_rate'])
    name += '_en_dim_past' + "-".join(map(str, event_proposal_model_param_dict['dim_past_list']))
    name += '_en_dim_curr' + "-".join(map(str, event_proposal_model_param_dict['dim_cur_list']))
    name += '_en_maxp{}'.format(max_proposals)
    print("GLOBAL NAME ", name)
    return name


def print_to_text_file_results(file_name, epoch, loss_dict, score_dict, iteration):
    if epoch == 0 and iteration == 0:
        header = ['epoch', 'iteration']
        header.extend([ele for ele in loss_dict.keys()])
        header.extend([ele for ele in score_dict.keys()])
        with open(file_name, 'a+') as fp:
            fp.write(",".join(header) + "\n")

    arr_scores_loss = [epoch, iteration]
    for ele in loss_dict.keys():
        if len(loss_dict[ele]) == 0:
            arr_scores_loss.appned(0)
        else:
            arr_scores_loss.append(loss_dict[ele][-1])
    for ele in score_dict.keys():
        if len(score_dict[ele]) == 0:
            arr_scores_loss.append(0)
        else:
            arr_scores_loss.append(score_dict[ele][-1])
    with open(file_name, 'a+') as fp:
        fp.write(",".join(map(str, arr_scores_loss)) + "\n")


def print_summary_to_file_results(file_name, epoch, loss_dict, score_dict):
    if epoch == 0:
        header = ['epoch']
        header.extend([ele for ele in loss_dict.keys()])
        header.extend([ele for ele in score_dict.keys()])
        with open(file_name, 'a+') as fp:
            fp.write(",".join(header) + "\n")

    arr_scores_loss = [epoch]
    for ele in loss_dict.keys():
        arr_scores_loss.append(loss_dict[ele])
    for ele in score_dict.keys():
        arr_scores_loss.append(score_dict[ele])
    with open(file_name, 'a+') as fp:
        fp.write(",".join(map(str, arr_scores_loss)) + "\n")


def print_tensorboard_results(epoch_writer, epoch, loss_dict, score_dict, iteration):
    label_base_loss = "epoch_losses_" + str(epoch)
    label_base_score = "epoch_scores_" + str(epoch)
    loss_dict_last = {}
    for ele in loss_dict.keys():
        if len(loss_dict[ele]) == 0:
            loss_dict_last[ele] = 0
        else:
            loss_dict_last[ele] = loss_dict[ele][-1]
    arr_dict_last = {}
    for ele in score_dict.keys():
        if len(score_dict[ele]) == 0:
            arr_dict_last[ele] = 0
        else:
            arr_dict_last[ele] = score_dict[ele][-1]
    epoch_writer.add_scalars(label_base_loss, loss_dict_last, iteration)
    epoch_writer.add_scalars(label_base_score, arr_dict_last, iteration)


def print_tensorboard_summary(epoch_writer, epoch, loss_dict, score_dict):
    label_base_loss = "summary_losses"
    label_base_score = "summary_scores"
    epoch_writer.add_scalars(label_base_loss, loss_dict, epoch)
    epoch_writer.add_scalars(label_base_score, score_dict, epoch)


def cal_recall_regress(p_network, p_gt_strt, p_gt_end, p_pred_strt, p_pred_len):
    iou_scores = network.cal_iou_one_dim(p_gt_strt, p_gt_end, p_pred_strt, p_pred_len)
    # print("Iou Scores = ", iou_scores)
    l_avg_iou = torch.mean(iou_scores.squeeze())

    recall = np.empty(iou_threshold.shape[0])
    # matches = valid_proposals iou_scores > threshold / total_valid_proposals
    for cidx, this_iou in enumerate(iou_threshold):
        num_gt_this_iou = torch.sum(iou_scores > this_iou)
        if iou_scores.shape[0] == 0:
            recall[cidx] = 0
        else:
            recall[cidx] = num_gt_this_iou.double() / len(iou_scores)

    return recall, l_avg_iou


def calculate_scores(trg, predicted):
    bleu_3_scores = []
    bleu_4_scores = []
    meteor_scores = []
    TRG_EOS_IDX = data_vocab.vocab.stoi[data_vocab.eos_token]
    for trgt_arr, predicted_arr in zip(trg, predicted):
        trg_sentence = ''
        for word_idx in trgt_arr[1:]:
            if word_idx == TRG_EOS_IDX:
                break
            trg_sentence = trg_sentence + " " + remove_nonascii(data_vocab.vocab.itos[word_idx])
        pred_sentence = ''
        predicted_arr_max = predicted_arr.argmax(1)
        for word_idx in predicted_arr_max[1:]:
            if word_idx == TRG_EOS_IDX:
                break
            pred_sentence = pred_sentence + " " + remove_nonascii(data_vocab.vocab.itos[word_idx])
        score_map = calculate_bleu_meteor_scores(pred_sentence, trg_sentence)
        bleu_3_scores.append(score_map['bleu_3'])
        bleu_4_scores.append(score_map['bleu_4'])
        meteor_scores.append(score_map['meteor'])
    return np.mean(np.array(bleu_3_scores)), np.mean(np.array(bleu_4_scores)), np.mean(np.array(meteor_scores))


def update_json_object(l_dense_cap_json, l_proposals_json, p_start, p_length, p_proposal,
                       sentences_output, filter_used, video_id):
    TRG_EOS_IDX = data_vocab.vocab.stoi[data_vocab.eos_token]
    bs = p_start.shape[0]
    filter2d = filter_used.view(bs, p_start.shape[1])
    row_index = 0
    all_indices = torch.arange(len(filter_used))
    map_sent = all_indices[filter_used == 1]
    map_sent_dict = {}
    for i, ele in enumerate(map_sent):
        map_sent_dict[ele.item()] = i
    for vid, proposal_start, proposal_length, proposal_score in zip(video_id, p_start, p_length, p_proposal):
        l_dense_cap_json['v_' + vid] = []
        l_proposals_json[vid] = []
        col_index = 0
        for p_s, p_l, p_score in zip(proposal_start, proposal_length, proposal_score):

            if filter2d[row_index][col_index] == 1:
                sentence = sentences_output[map_sent_dict[row_index * p_start.shape[1] + col_index]]
                predicted_arr_max = sentence.argmax(1)
                pred_sentence = ''
                for word_idx in predicted_arr_max[1:]:
                    if word_idx == TRG_EOS_IDX:
                        break
                    pred_sentence = pred_sentence + " " + remove_nonascii(data_vocab.vocab.itos[word_idx])

                p_s_print = p_s.item()
                p_end_print = p_s.item() + p_l.item()
                p_score_print = p_score.item()
                l_dense_cap_json['v_' + vid].append(
                    {'sentence': pred_sentence,
                     'timestamp': [p_s_print, p_end_print]})

                l_proposals_json[vid].append(
                    {'segment': [p_s_print, p_end_print],
                     'score': p_score_print})
            col_index += 1
        row_index += 1


def one_pass(p_network, p_loader, p_optim, p_mse_criterion, p_bce_criterion, p_criterion_sentence, p_is_train, p_epoch,
             p_proposal_criterion):
    # Input: Network and Optimizer
    # tensorboard Output: Averge accuracy , Avergae loss in the pass
    dense_cap_all = None
    if args.p_json:
        densecap_result = {}
        proposal_result = {}
        proposal_all = {'version': "VERSION 1.3", 'external_data': {'used': 'true',
                                                                    'details': 'Temporal Aggregates for localization'}}
        dense_cap_all = {'version': 'VERSION 1.0', 'external_data': {'used': 'true',
                                                                     'details': 'Temporal Aggregates for localization '
                                                                                'and sentences'}}
    stage = 'Validation'
    if p_is_train:
        stage = 'Training'
        p_network.train()
    else:
        p_network.eval()

    epoch_writer = None
    if is_tensorboard is True:
        epoch_writer = SummaryWriter(os.path.join(args.tb_logs, 'epoch_' + str(p_epoch) + stage + '_summary'))
    l_file_name = None
    if print_results is True:
        l_file_name = os.path.join(args.text_logs, 'epoch_' + str(p_epoch) + stage + '_summary')
    acc_one_pass = []
    loss_one_pass = []
    other_losses = {}
    other_acc = {}

    other_losses['start_loss'] = []
    other_losses['len_loss'] = []
    other_losses['vocab_loss'] = []
    other_losses['proposal_score_loss'] = []
    other_losses['logit_loss'] = []
    other_losses['total_loss'] = []
    other_acc['recall_30'] = []
    other_acc['recall_50'] = []
    other_acc['recall_70'] = []
    other_acc['recall_90'] = []
    other_acc['avg_recall'] = []
    other_acc['avg_iou'] = []
    other_acc['bleu_3'] = []
    other_acc['bleu_4'] = []
    other_acc['meteor'] = []

    for i, ele in enumerate(iter(p_loader)):
        video_id = ele['id']
        recent_features = []
        for ele_r in ele['recent_features']:
            recent_features.append(ele_r.type(torch.FloatTensor).to(device))

        past_features = []
        for ele_p in ele['past_features']:
            past_features.append(ele_p.type(torch.FloatTensor).to(device))

        sentences = ele['sentences'].type(torch.LongTensor).to(device)

        # out['label']  # [[27, 2][27, 2]] , start index, end_index of current time divison
        # out['current_divison_timings'] # [[31],[51]], time indexing of indexes, make (arr[index] + arr[index+1]) / 2
        # out['actual_timings'] # [27,2], actual timings given in the data
        # calculate the average of index, index + 1 for 30 divisions and 50 divsions, 
        # and then calculate the average of 30 and 50 timings

        labels = []
        for ele_l in ele['label']:
            labels.append(ele_l.type(torch.LongTensor).to(device))  # ele_l = [bs, max_prop, 2]

        bin_proposal_list = []
        for ele_l in ele['bin_proposal']:
            bin_proposal_list.append(ele_l.type(torch.FloatTensor).to(device))  # ele_l = [bs, max_prop, 30]

        current_divison_timings = []
        for cur_div in ele['current_divison_timings']:
            current_divison_timings.append(
                cur_div.type(torch.FloatTensor).to(device))  # current_divison_timings = [bs, 31]

        gt_truth_timings = ele['actual_timings'].type(torch.FloatTensor).to(device)  # gt_truth_timings = [bs, max_prop, 2]
        duration = ele['duration'].type(torch.FloatTensor).to(device)

        with torch.set_grad_enabled(p_is_train):
            sentences_sent = None
            if p_is_train:
                sentences_sent = sentences
                # input_sentences, x_past_actual_all, x_curr_actual_all, labels, is_train, teacher_forcing_ratio
            proposal_score, outputs, filter_used, attn_weights_list = p_network(sentences_sent, past_features,
                                                                                recent_features, gt_truth_timings,
                                                                                p_is_train, teacher_forcing_ratio,
                                                                                duration)

            # proposal_score, encoded_outputs = p_network(past_features, recent_features)
            # proposal_score = [bs, max_proposals, 3] 0- start ratio, 1- end ratio, 2- probability score
            start_score = proposal_score[:, :, 0].squeeze()
            duration_expanded = duration.unsqueeze(1).expand_as(start_score)
            predicted_start_proposal = start_score * duration_expanded
            predicted_start_proposal_flt = predicted_start_proposal.squeeze().contiguous().view(-1, 1)

            length_score = proposal_score[:, :, 1].squeeze()
            predicted_len_proposal = (duration_expanded - predicted_start_proposal) * length_score
            predicted_len_proposal_flt = predicted_len_proposal.view(-1, 1)

            predicted_scores = proposal_score[:, :, 2].squeeze().contiguous().view(-1, 1).squeeze()

            gt_truth_timings_start = gt_truth_timings[:, :, 0].squeeze().view(-1, 1)
            gt_truth_timings_end = gt_truth_timings[:, :, 1].squeeze().view(-1, 1)
            gt_truth_timings_length = gt_truth_timings_end - gt_truth_timings_start
            ground_truth_proposals_valid = torch.gt(gt_truth_timings_end, gt_truth_timings_start).squeeze().type(
                torch.FloatTensor).to(device)

            filter_proposals = gt_truth_timings_length > 0
            if p_is_train is False:
                filter_proposals = predicted_scores > threshold

            gt_truth_timings_start_filtered = gt_truth_timings_start[filter_proposals]
            predicted_start_proposal_flt_filtered = predicted_start_proposal_flt[filter_proposals]
            predicted_len_proposal_flt_filtered = predicted_len_proposal_flt[filter_proposals]
            gt_truth_timings_length_filtered = gt_truth_timings_length[filter_proposals]
            gt_truth_timings_end_filtered = gt_truth_timings_end[filter_proposals]

            loss_start = p_mse_criterion(predicted_start_proposal_flt_filtered, gt_truth_timings_start_filtered)
            loss_length = p_mse_criterion(predicted_len_proposal_flt_filtered, gt_truth_timings_length_filtered)
            loss_valid_proposals = p_bce_criterion(predicted_scores, ground_truth_proposals_valid)

            loss_encoding_proposal = torch.tensor(0, dtype=torch.float32).to(device)
            for pred, target in zip(attn_weights_list, bin_proposal_list):
                filtered_t = target[target.sum(dim=2) > 0]
                filtered_p = pred[target.sum(dim=2) > 0]
                loss_encoding_proposal = loss_encoding_proposal + proposal_criterion(filtered_p, filtered_t)

            vocab_loss = torch.tensor(0, dtype=torch.float32).to(device)
            trg = None
            if outputs is not None:
                output_dim = outputs.shape[-1]
                all_sentences = sentences.view(-1, 1, sentences.shape[2]).squeeze(1)
                trg = all_sentences[filter_used]
                outputs_flt = outputs[:, 1:].contiguous().view(-1, 1, output_dim).squeeze()
                trg_flt = trg[:, 1:].contiguous().view(-1, 1).squeeze()

                vocab_loss = p_criterion_sentence(outputs_flt, trg_flt)

            total_loss = vocab_loss + loss_start + loss_length + loss_valid_proposals + loss_encoding_proposal

        if p_is_train is True:
            p_optim.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(p_network.parameters(), clip)
            p_optim.step()

        with torch.no_grad():
            # recall_scores = cal_recall(pred_list, labels, current_divison_timings, gt_truth_timings, proposals_score_list)
            recall_scores, avg_iou = cal_recall_regress(p_network, gt_truth_timings_start_filtered, gt_truth_timings_end_filtered,
                                                        predicted_start_proposal_flt_filtered,
                                                        predicted_len_proposal_flt_filtered)

            other_losses['start_loss'].append(loss_start.item())
            other_losses['len_loss'].append(loss_length.item())
            other_losses['vocab_loss'].append(vocab_loss.item())
            other_losses['proposal_score_loss'].append(loss_valid_proposals.item())
            other_losses['logit_loss'].append(loss_encoding_proposal.item())
            other_losses['total_loss'].append(total_loss.item())
            other_acc['recall_30'].append(recall_scores[0].item())
            other_acc['recall_50'].append(recall_scores[1].item())
            other_acc['recall_70'].append(recall_scores[2].item())
            other_acc['recall_90'].append(recall_scores[3].item())
            other_acc['avg_recall'].append(np.mean(recall_scores))
            other_acc['avg_iou'].append(avg_iou.item())

            if i % 100 == 0:
                print("Iter: {:2.2f}. ".format(i / len(p_loader)))
                print("Start Loss : {:5.2f}. ".format(loss_start.item()))
                print("Length Loss : {:5.2f}. ".format(loss_length.item()))
                print("Valid proposal loss : {:5.2f}. ".format(loss_valid_proposals.item()))
                print("Encoded proposal loss : {:5.2f}. ".format(loss_encoding_proposal.item()))
                print("Vocab loss: {:3.2f}. ".format(vocab_loss.item()),
                      "{:20s} Total Loss: {:3.2f}. ".format(stage, total_loss.item()),
                      )
                for iou_index, iou in enumerate(iou_threshold):
                    print("Recall for iou_threshold: {:2.2f} is {:2.2f}".format(iou, recall_scores[iou_index]))
                print("Recall for all iou_avg is : {:2.2f}".format(np.mean(recall_scores)))
                print("Avegrae recall {:3.2f}".format(avg_iou))

                if outputs is not None:
                    rand_num = np.random.randint(0, outputs.shape[0])
                    for s_i, sentence in enumerate(trg[rand_num:rand_num + 1]):
                        trg_sentence = [remove_nonascii(data_vocab.vocab.itos[i]) for i in sentence]
                        print("Target sentence: ", s_i, "is ", " ".join(trg_sentence))
                    for s_i, sentence_prob in enumerate(outputs[rand_num:rand_num + 1]):
                        sentence = sentence_prob.argmax(1)
                        trg_sentence = [remove_nonascii(data_vocab.vocab.itos[i]) for i in sentence]
                        print("Source sentence: ", s_i, "is ", " ".join(trg_sentence))
                    if is_calculate_scores is True:
                        bleu3_arr, bleu4_arr, meteor_arr = calculate_scores(trg, outputs)
                        other_acc['bleu_3'].extend(bleu3_arr)
                        other_acc['bleu_4'].extend(bleu4_arr)
                        other_acc['meteor'].extend(meteor_arr)

            if is_tensorboard is True:
                print_tensorboard_results(epoch_writer, p_epoch, other_losses, other_acc, i)
            if print_results is True:
                print_to_text_file_results(l_file_name, p_epoch, other_losses, other_acc, i)
            acc = np.mean(recall_scores)
            acc_one_pass.append(acc)
            loss_one_pass.append(total_loss.item())
            if dense_cap_all is not None:
                update_json_object(densecap_result, proposal_result, predicted_start_proposal, predicted_len_proposal,
                                   proposal_score[:, :, 2].squeeze().contiguous(), outputs, filter_used, video_id)

    avg_acc = sum(acc_one_pass) / len(acc_one_pass)
    avg_loss = sum(loss_one_pass) / len(loss_one_pass)
    if is_tensorboard is True:
        epoch_writer.close()
    loss_dict = {}
    acc_dict = {}
    for ele in other_losses.keys():
        loss_dict[ele] = np.mean(np.array(other_losses[ele]))
    for ele in other_acc.keys():
        acc_dict[ele] = np.mean(np.array(other_acc[ele]))
    if dense_cap_all is not None:
        dense_cap_all['results'] = densecap_result
        with open(os.path.join(args.json_logs, 'densecap_validation.json'), 'w') as f:
            json.dump(dense_cap_all, f)
        proposal_all['results'] = proposal_result
        with open(os.path.join(args.json_logs, 'proposal_validation.json'), 'w') as f:
            json.dump(proposal_all, f)

    return avg_acc, avg_loss, loss_dict, acc_dict


def multiple_pass(p_network, p_optim, p_trainloader, p_valloader, p_name, l_mse_criterion, l_bse_criterion,
                  l_criterion_sentence, l_proposal_criterion):
    # Input: p_network, p_optim
    # Output: Output a chart accuracy 
    global g_start_epoch, g_best_perf
    all_epochs_acc = []
    all_epochs_loss = []
    pos_weight = torch.tensor([8, 2]).type(torch.FloatTensor).to(device)
    writer_train = None
    writer_valid = None
    l_file_name_train = None
    l_file_name_valid = None

    if print_results is True:
        l_file_name_train = os.path.join(args.text_logs, 'summary_train')
        l_file_name_valid = os.path.join(args.text_logs, 'summary_validation')
    if is_tensorboard is True:
        writer_train = SummaryWriter(os.path.join(args.tb_logs, 'summary_train'))
        writer_valid = SummaryWriter(os.path.join(args.tb_logs, 'summary_validation'))

    for epoch in range(g_start_epoch, epochs):
        start_time = time.time()
        # factor = learning_rate_mod_factor(epoch)
        # for i, g in enumerate(p_optim.param_groups):
        #    g['lr'] = learning_rate * factor
        #    print("Learning rate for param %d is currently %f" %(i, g['lr']))

        avg_acc, avg_loss, other_loss, other_acc = one_pass(p_network, p_trainloader, p_optim,
                                                            l_mse_criterion, l_bse_criterion,
                                                            l_criterion_sentence, True, epoch, l_proposal_criterion)
        print('\033[92m',
              "Epoch: {:d}. ".format(epoch),
              "Training Total Loss: {:.2f}. ".format(avg_loss),
              "Training Accuracy: {:.2f}% ".format(avg_acc),
              '\033[0m')
        if is_tensorboard is True:
            print_tensorboard_summary(writer_train, epoch, other_loss, other_acc)
        if print_results is True:
            print_summary_to_file_results(l_file_name_train, epoch, other_loss, other_acc)

        val_accuracy, val_loss, other_loss_val, other_acc_val = one_pass(p_network, p_valloader, p_optim,
                                                                         l_mse_criterion, l_bse_criterion,
                                                                         l_criterion_sentence,
                                                                         False, epoch, l_proposal_criterion)
        if g_best_perf < val_accuracy:
            g_best_perf = val_accuracy
            l_is_best = True
        else:
            l_is_best = False

        end_time = time.time()

        with open(model_file_path + '/' + p_name + '.txt', 'a') as f:
            f.write("%d - %0.2f\n" % (epoch + 1, val_accuracy))

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print('\033[92m',
              "Epoch: {:d}. ".format(epoch),
              "Val Total Loss: {:.2f}. ".format(val_loss),
              "Val Accuracy: {:.2f}% ".format(val_accuracy),
              "Best Val Accuracy: {:.2f}% ".format(g_best_perf),
              '\033[0m')

        # save checkpoint at the end of each train/val epoch
        save_model(p_network, p_name, epoch + 1, val_accuracy, g_best_perf, p_optim, is_best=l_is_best)

        if is_tensorboard is True:
            print_tensorboard_summary(writer_valid, epoch, other_loss_val, other_acc_val)
        if print_results is True:
            print_summary_to_file_results(l_file_name_valid, epoch, other_loss_val, other_acc_val)


        all_epochs_loss.append(val_loss)
        all_epochs_acc.append(val_accuracy)

    with open(model_file_path + '/' + p_name + '.txt', 'a') as f:
        f.write("%d - %0.2f\n" % (epoch + 1, g_best_perf))
    # Plotting the accuracy and loss graphs
    print("all_epochs_acc = \n", all_epochs_acc)
    print("all epochs loss = \n", all_epochs_loss)
    if writer_train:
        writer_train.close()
    if writer_valid:
        writer_valid.close()


################ Model Name Creation######################
g_name = make_model_name()

if is_tensorboard:
    args.tb_logs = os.path.join(args.tb_logs, g_name)
    if not os.path.exists(args.tb_logs):
        os.mkdir(args.tb_logs)
        print("Dumping the tensor board results files in {} directory".format(args.tb_logs))

if print_results:
    args.text_logs = os.path.join(args.text_logs, g_name)
    if not os.path.exists(args.text_logs):
        os.mkdir(args.text_logs)
        print("Dumping the text results files in {} directory".format(args.text_logs))

if args.p_json:
    args.json_logs = os.path.join(args.json_logs, g_name)
    if not os.path.exists(args.json_logs):
        os.mkdir(args.json_logs)
        print("Dumping the json files in {} directory".format(args.json_logs))

#############################Declaring dataset######################################
train_dataset = activity_net.ActivityNet(json_data, duration_file, os.path.join(features_base_dir + 'training'),
                                         feature_names,
                                         event_proposal_model_param_dict, encoding_proposal_model_param_dict,
                                         captioning_proposal_model_param_dict, data_vocab)
val_dataset = activity_net.ActivityNet(json_data, duration_file, os.path.join(features_base_dir + 'validation'),
                                       feature_names,
                                       event_proposal_model_param_dict, encoding_proposal_model_param_dict,
                                       captioning_proposal_model_param_dict, data_vocab)
trainloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=workers,
                         pin_memory=True, drop_last=True, shuffle=True, collate_fn=activity_net.my_collate)
valloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=workers,
                       pin_memory=True, drop_last=True, shuffle=False, collate_fn=activity_net.my_collate)

############################Declaring Models###########################################
# if args.no_tab_no_lstm:
# 
# else:
if args.no_tab:
    from my_models import to_tab_network
    encoding_network = to_tab_network.OverallNetwork(non_local_params, event_proposal_model_param_dict,
                                              encoding_proposal_model_param_dict,
                                              captioning_proposal_model_param_dict, feature_dim, linear_dim, embed_dim,
                                              vocab_size,
                                              pad_idx, sentence_train_start, data_vocab)
else:
    encoding_network = network.OverallNetwork(non_local_params, event_proposal_model_param_dict,
                                              encoding_proposal_model_param_dict,
                                              captioning_proposal_model_param_dict, feature_dim, linear_dim, embed_dim,
                                              vocab_size,
                                              pad_idx, sentence_train_start, data_vocab)
# encoding_network = network.EncodingProposal(non_local_params, event_proposal_model_param_dict, linear_dim,
# feature_dim)

encoding_network.to(device)

count_params = sum([p.numel() for p in encoding_network.parameters()])
print("Number of parameters in network ", count_params / 1e6)

TRG_PAD_IDX = data_vocab.vocab.stoi[data_vocab.pad_token]
g_criterion_sentence = nn.CrossEntropyLoss(ignore_index=TRG_PAD_IDX)
g_mse_criterion = nn.MSELoss()
g_bse_criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([0.85]))
g_bse_criterion = g_bse_criterion.to(device)
g_bse_proposal_criterion = nn.BCELoss()

# optimizer = torch.optim.Adam(encoding_network.parameters(), lr=learning_rate, betas=(0.99, 0.999))
optimizer = torch.optim.Adam([{'params': encoding_network.event_proposal.parameters(), 'lr': learning_rate},
                              {'params': encoding_network.encode_proposal.parameters(), 'lr': learning_rate},
                              {'params': encoding_network.decode_proposal.parameters(), 'lr': args.dec_lr}], lr=learning_rate,
                             betas=(0.99, 0.999))

# optimizer = torch.optim.SGD(encoding_network.parameters(), lr=learning_rate, momentum=0.9)
# avg_acc, avg_loss = train_one_pass(encoding_network, optimizer, trainloader)

##############Train and validate multple pass#####
if is_run_train_valid is True:
    # adding the last checkpoint to resume training
    if resume is True:
        g_start_epoch, curr_perf, g_best_perf = load_checkpoint(encoding_network, g_name, optimizer)
        print("Loaded checkpoint with epoch {}, curr_perf {} and best perf {}".format(g_start_epoch, curr_perf,
                                                                                      g_best_perf))
        g_start_epoch += 1
    multiple_pass(encoding_network, optimizer, trainloader, valloader, g_name, g_mse_criterion, g_bse_criterion,
                  g_criterion_sentence, g_bse_proposal_criterion)

###################Finding the learning rate###########

# TRG_PAD_IDX = data_vocab.vocab.stoi[data_vocab.pad_token]
# l_criterion_sentence = nn.CrossEntropyLoss(ignore_index = TRG_PAD_IDX)
# l_mse_criterion = nn.MSELoss()
# l_bse_criterion = nn.BCELoss()
# from lr_finder import LRFinder
# lr_finder = LRFinder(model=encoding_network, optimizer=optimizer, criterion=[l_mse_criterion, l_bse_criterion, l_criterion_sentence], device=device)
# lr_finder.range_test(trainloader, end_lr=10, num_iter=500, step_mode="exp")
# lr_finder.plot(fname='lr_probing.pdf')


#######################Validation and json file creation##########################
if is_run_valid_only is True:
    start_time = time.time()
    _, curr_perf, g_best_perf = load_checkpoint(encoding_network, g_name, optimizer, args.pick_best)
    val_accuracy, val_loss, other_loss_val, other_acc_val = one_pass(encoding_network, valloader, optimizer,
                                                                     g_mse_criterion, g_bse_criterion,
                                                                     g_criterion_sentence, False, 0,
                                                                     g_bse_proposal_criterion)
    end_time = time.time()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    print(f'Total Time for evaluation : {epoch_mins}m {epoch_secs}s')
    print(other_loss_val)
    print(other_acc_val)

