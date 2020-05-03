import os
import json
import h5py
import numpy as np
import csv
from collections import defaultdict
import math
import multiprocessing
import pickle
from random import shuffle

import torch
import torchtext
from torch.utils.data import Dataset

from torch.utils.data.dataloader import default_collate


def get_vocab_and_sentences(dataset_file, max_length=20):
    # build vocab and tokenized sentences
    text_proc = torchtext.data.Field(sequential=True, init_token='<init>',
                                     eos_token='<eos>', tokenize='spacy',
                                     lower=True, batch_first=True,
                                     fix_length=max_length)
    train_val_sentences = []

    with open(dataset_file, 'r') as data_file:
        data_all = json.load(data_file)
    data = data_all['database']

    nsentence = {}
    nsentence['training'] = 0
    nsentence['validation'] = 0
    ntrain_videos = 0
    max_proposal = 0
    for vid, val in data.items():
        anns = val['annotations']
        if len(anns) > max_proposal:
            max_proposal = len(anns)
        split = val['subset']
        if split == 'training':
            ntrain_videos += 1
        if split in ['training', 'validation']:
            for ind, ann in enumerate(anns):
                ann['sentence'] = ann['sentence'].strip()
                train_val_sentences.append(ann['sentence'])
                nsentence[split] += 1

    sentences_proc = list(map(text_proc.preprocess, train_val_sentences)) # build vocab on train and val
    text_proc.build_vocab(sentences_proc, min_freq=5)
    print('# of words in the vocab: {}'.format(len(text_proc.vocab)))
    print('# of sentences in training: {}, # of sentences in validation: {}'.format(
            nsentence['training'], nsentence['validation']
        ))
    print('# of training videos: {}'.format(ntrain_videos))
    print('Max size of proposal: {}'.format(max_proposal))
    return text_proc, data, max_proposal


def get_frames_per_sec(dataset, duration_file):
    # Returns a dictionary containing the number of frames per sec for a particular videos{'vid_name':30,..}
    frame_per_second = {}
    sampling_sec = 0.5 # hard coded, only support 0.5
    with open(duration_file) as f:
        if dataset == 'anet':
            for line in f:
                vid_name, vid_dur, vid_frame = [l.strip() for l in line.split(',')]
                frame_per_second[vid_name] = float(vid_dur)*int(float(vid_frame)*1./int(float(vid_dur))*sampling_sec)*1./float(vid_frame)
                frame_per_second[vid_name] = 1.0 / float(frame_per_second[vid_name])
            frame_per_second['_0CqozZun3U'] = sampling_sec # a missing video in anet
            frame_per_second['_0CqozZun3U'] = 1.0 / float(frame_per_second['_0CqozZun3U'])
        elif dataset == 'yc2':
            for line in f:
                vid_name, vid_dur, vid_frame = [l.strip() for l in line.split(',')]
                frame_per_second[vid_name] = float(vid_dur)*math.ceil(float(vid_frame)*1./float(vid_dur)*sampling_sec)*1./float(vid_frame)
                # for yc2
                frame_per_second[vid_name] = 1.0 / float(frame_per_second[vid_name])
        else:
            raise NotImplementedError
    return frame_per_second


def get_max_pooled_features(env, frame_indices):
    list_data = []
    for kkl in range(len(frame_indices) - 1):
        cur_start = max(0, np.floor(frame_indices[kkl]).astype('int') - 1)
        cur_end   = min((np.ceil(frame_indices[kkl + 1]).astype('int') - 1), env.shape[0] - 1)
        list_frames = list(range(cur_start, cur_end + 1))
        if len(list_frames) == 0:
#             print(frame_indices)
#             print("list_frames = 0. Start and end encountered =", cur_start, cur_end + 1)
#             print(env.shape)
            if cur_end <= cur_start:
                cur_start = cur_end - 2
            if len(list_data) == 0:
                print("DATA returned is NONE", frame_indices, cur_start, cur_end)
                return None
            list_data.append(list_data[-1])
            continue
        pool_list = []
        for name in list_frames:
            data = env[name]
            pool_list.append(data)
        pool_ndarr = np.array(pool_list)
        max_pool = np.max(pool_ndarr, 0)
        list_data.append(max_pool.squeeze())
   
    if len(list_data) == 0:
        print("ERROR: None of the partitions were non-zero")
#         print(frame_indices)
#         print(env.shape)
        return None 
    list_data  =  np.stack(list_data)
    # print(list_data.shape)
    return list_data


def read_c3d_feature(file):
    for key in file.keys():
        feature = file[key + '/c3d_features'][:]
    return feature


def read_representations(recent_frames, past_frames, base_root, feature_names, video_name):
    """ Reads a set of representations, given their frame names and an LMDB environment.
    Applies a transformation to the features if provided"""
    current = []
    past    = []

    recent_features_arr = []  
    past_features_arr   = []
    # print("Shape of recent frame ", len(recent_frames)) #, "Shape of 0th frame", recent_frames[0].shape)
    # print("Shape of past frame ", len(past_frames)) # , "Shape of 0th frame", past_frames[0].shape)
    for e in feature_names:
        if e == 'c3d':
            f = h5py.File('/mnt/data/captioning_dataset/activity_net/data/c3d_features/sub_activitynet_v1-3.c3d.hdf5',
                          'r')
            video_numpy_arr = f['v_' + video_name + '/c3d_features'][:]
        else:
            if not os.path.exists(os.path.join(base_root, video_name + e + ".npy")):
                print("File does not exists", os.path.join(base_root, video_name + e + ".npy"))
                return None
            if not os.path.exists(os.path.join(base_root, video_name + '_resnet.npy')):
                print("File does not exists", os.path.join(base_root, video_name + e + ".npy"))
                return None
            video_numpy_arr = np.load(os.path.join(base_root, video_name + e + ".npy"))
            flow_numpy_arr = np.load(os.path.join(base_root, video_name + '_resnet.npy'))

            # if np.sum(np.isnan(video_numpy_arr)) != 0 or np.sum(np.isnan(flow_numpy_arr)) != 0:
            #     return None
            video_numpy_arr = np.hstack((video_numpy_arr, flow_numpy_arr))

        recent_env_arr = []
        past_env_arr   = []
        for i in range(len(recent_frames)):
            recent_features = get_max_pooled_features(video_numpy_arr, recent_frames[i])
            if recent_features is not None:
                recent_env_arr.append(recent_features)

        if len(recent_env_arr) == 0:
            print("Skipping: None of current frames is 1")
            return None, None
        elif len(recent_env_arr) < len(recent_frames):
            random_arr = np.random.randint(0, len(recent_env_arr), len(recent_frames) - len(recent_env_arr))
            for nums in random_arr:
                recent_env_arr.append(recent_env_arr[nums])       
       
        #skipped_frames = []
        for i in range(len(past_frames)):
            past_features = get_max_pooled_features(video_numpy_arr, past_frames[i])
            if past_features is not None:
                past_env_arr.append(past_features)

        if len(past_env_arr) == 0:
            print("Skipping: None of past frames is 1")
            return None, None
        elif len(past_env_arr) < len(past_frames):
            random_arr = np.random.randint(0, len(past_env_arr), len(past_frames) - len(past_env_arr))
            for nums in random_arr:
                past_env_arr.append(past_env_arr[nums])       

        recent_features_arr.append(recent_env_arr)
        past_features_arr.append(past_env_arr)

    recent_features_arr = np.concatenate(recent_features_arr, axis=-1)
    past_features_arr   = np.concatenate(past_features_arr, axis=-1)
    # print("Recent features size ", recent_features_arr.shape)
    # print("Past features size ", past_features_arr.shape)

    for i in range(len(recent_features_arr)):
        # print("present features array", (recent_features_arr[i].shape))
        current.append(recent_features_arr[i])

    for i in range(len(past_features_arr)):
        # print("past features array", (past_features_arr[i].shape))
        past.append(past_features_arr[i])

    return current, past


def read_data(recent_frames_intervals, past_frames_intervals, base_root, feature_names, video_name):
    """A wrapper form read_representations to handle loading from more environments.
    This is used for multimodal data loading (e.g., RGB + Flow)"""
    # if env is a list
    if len(feature_names) > 1:
        # read the representations from all environments
        raise NotImplementedError
    else:
        # otherwise, just read the representations
        return read_representations(recent_frames_intervals, past_frames_intervals, base_root, feature_names, video_name)

def my_collate(batch):
    "Puts each data field into a tensor with outer dimension batch size"
    batch = list(filter (lambda x:x is not None, batch))
    return default_collate(batch)


class ActivityNet(Dataset):
    def __init__(self, json_data_obj, duration_file, base_dir, feature_names,
                 event_proposal_params, encod_proposal_params, caption_proposal_params, text_proc):
        """
        Args:
            base_dir (string): Directory containing features
            is_train (bool): training Dataset or validation dataset
        """
        self.json_data_obj = json_data_obj
        self.discarded_items = []
        
        self.frame_per_second = get_frames_per_sec('anet', duration_file)
        self.event_proposal_params = event_proposal_params
        self.encod_proposal_params = encod_proposal_params
        self.caption_proposal_param = caption_proposal_params
        self.feature_names = feature_names
        self.base_dir = base_dir
        self.is_train = False
        self.text_proc = text_proc
        if 'training' in base_dir:
            self.is_train = True
            
        self.final_list, self.sentences_indices, self.sentence_keys = self.create_final_items()
        print("Length of final list = ", len(self.final_list))
        print("Length of discarded items list = ", len(self.discarded_items))

    def create_final_items(self):
        full_list = []
        labels_dic = dict()
        count = 0
        sentences_list = []
        sentences_keys = {}
        count_sentence = 0
        for video_name, val in self.json_data_obj.items():
            split = val['subset']
            
            if self.is_train and split != 'training':
                continue
            if not os.path.exists(os.path.join(self.base_dir, video_name + "_bn.npy")):
                self.discarded_items.append(video_name)
                # print("File does not exists", os.path.join(self.base_dir, video_name + "_bn.npy"))
                continue
            if not os.path.exists(os.path.join(self.base_dir, video_name + "_resnet.npy")):
                self.discarded_items.append(video_name)
                # print("File does not exists", os.path.join(self.base_dir, video_name + "_resnet.npy"))
                continue
            if not self.is_train and split != 'validation':
                continue
            if video_name not in self.frame_per_second:
                print(video_name)
                continue
            duration = val['duration']
            segments = val['annotations']
            event_proposal = {}
            past_divisons_frames = []
            for past_div in self.event_proposal_params['dim_past_list']:
                if self.feature_names[0] == 'c3d':
                    past_divisons_frames.append(
                        np.linspace(0, duration, past_div + 1) * self.frame_per_second[video_name] * 3 / 2)
                else:
                    past_divisons_frames.append(np.linspace(0, duration, past_div + 1) * self.frame_per_second[video_name])
            cur_divisons = []
            cur_div_frames = []
            pos_proposals = []
            binary_proposals = []
            act_proposals = np.zeros((self.event_proposal_params['max_proposals'], 2), dtype=np.float32)
            for cur_div in self.event_proposal_params['dim_cur_list']:
                if self.feature_names[0] == 'c3d':
                    cur_divisons.append(np.linspace(0, duration, cur_div + 1) * 3 / 2)
                else:
                    cur_divisons.append(np.linspace(0, duration, cur_div + 1))
                cur_div_frames.append(cur_divisons[-1] * self.frame_per_second[video_name])
                pos_proposals.append(np.zeros((self.event_proposal_params['max_proposals'], 2), dtype=np.int64))
                binary_proposals.append(np.zeros((self.event_proposal_params['max_proposals'], cur_div), dtype=np.bool))

            event_proposal['cur_divisons_frames'] = cur_div_frames
            # Number of current divisons * Number of splits in 1 current div
            event_proposal['past_divisons_frames'] = past_divisons_frames
            # Number of past divisons * Number of splits in 1 past div
            event_proposal['duration'] = duration # Duration of the videos
            
            # Marking true in positive proposals from the ground truth segments

            for row_index, ele in enumerate(segments):
                proposal_dur_start = ele['segment'][0]
                proposal_dur_end = ele['segment'][1]
                act_proposals[row_index][0] = proposal_dur_start
                act_proposals[row_index][1] = proposal_dur_end
                # print("Segment = ", proposal_dur_start, "," , proposal_dur_end)
                element_sent = ele['sentence']
                sentences_list.append(element_sent)
                sentences_keys[video_name + "_" + str(row_index)] = count_sentence
                count_sentence += 1
                for div, proposal in enumerate(pos_proposals):
                    flag = 0
                    # print("Current divisions = ", cur_divisons[div])
                    for index, st_dur in enumerate(cur_divisons[div][:-1]):
                        end_dur = cur_divisons[div][index + 1]
                        if (st_dur <= proposal_dur_start) and (proposal_dur_start <= end_dur):
                            flag = 1
                            proposal[row_index][0] = index + 1
                        binary_proposals[div][row_index][index] = flag
                        if (flag == 1) and (st_dur <= proposal_dur_end) and (proposal_dur_end <= end_dur):
                            proposal[row_index][1] = index + 1
                            flag = 0

            event_proposal['bin_proposals'] = binary_proposals
            event_proposal['pos_proposals'] = pos_proposals
            # Proposal of all current LSTM output i.e Max Proposals * Number of LSTM outs
            event_proposal['actual_timings'] = act_proposals
            event_proposal['time_proposals'] = cur_divisons
            full_list.append((video_name, event_proposal))
            count += 1
            # if count > 400:
            #    break
        train_sentences = list(map(self.text_proc.preprocess, sentences_list))
        sentence_idx = self.text_proc.numericalize(self.text_proc.pad(train_sentences), device=0).cpu().numpy()  # put in memory
        print("Shape of sentence idx ", sentence_idx.shape, "Size of sentence idx", sentence_idx.size)
        if sentence_idx.shape[0] != len(train_sentences):
                raise Exception("Error in numericalize sentences")
        return full_list, sentence_idx, sentences_keys
    
    def get_video_annotation(self, video_name):
        return self.json_data_obj[video_name]

    def get_item(self, idx):
        video_name, event_proposal_obj = self.final_list[idx]
        out = {'id': video_name}
        # read representations for past frames
        if len(self.feature_names) > 1:
            print("Not implemented error")
            raise NotImplementedError
        else:
            
            anno = self.get_video_annotation(video_name)
            frame_per_sec = self.frame_per_second[video_name]
            out['recent_features'], out['past_features'] = read_data(event_proposal_obj['cur_divisons_frames'], 
                                                                     event_proposal_obj['past_divisons_frames'], 
                                                                     self.base_dir, self.feature_names, video_name)
            out['sentences'] = np.ones((self.event_proposal_params['max_proposals'], 
                                        self.caption_proposal_param['max_sentence_len']), dtype=np.float64) * \
                                        self.text_proc.numericalize([['<pad>']])[0][0].item()
            for index in range(self.event_proposal_params['max_proposals']):
                key = video_name + '_' + str(index)
                if key in self.sentence_keys:
                    out['sentences'][index,:] = self.sentences_indices[self.sentence_keys[video_name + '_' + str(index)]]
                
        if out['recent_features'] is None and out['past_features'] is None:
            print("Discarded videos =", video_name)
            return None

        # get the label of the current sequence
        out['bin_proposal'] = event_proposal_obj['bin_proposals']
        out['label'] = event_proposal_obj['pos_proposals']
        out['current_divison_timings'] = event_proposal_obj['time_proposals']
        out['actual_timings'] = event_proposal_obj['actual_timings']
        out['duration'] = event_proposal_obj['duration']
        return out

    def __len__(self):
        return len(self.final_list)

    def __getitem__(self, idx):
        return self.get_item(idx)
