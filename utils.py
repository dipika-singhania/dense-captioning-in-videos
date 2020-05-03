"""
 Copyright (c) 2018, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.bleu.bleu import Bleu

import json
import urllib.request

import numpy as np

API = 'http://ec2-52-11-11-89.us-west-2.compute.amazonaws.com/challenge17/api.py'

def get_blocked_videos(api=API):
    api_url = '{}?action=get_blocked'.format(api)
    response = urllib.request.urlopen(api_url)
    return json.loads(response.read().decode('utf-8'))

def interpolated_prec_rec(prec, rec):
    """Interpolated AP - VOCdevkit from VOC 2011.
    """
    mprec = np.hstack([[0], prec, [0]])
    mrec = np.hstack([[0], rec, [1]])
    for i in range(len(mprec) - 1)[::-1]:
        mprec[i] = max(mprec[i], mprec[i + 1])
    idx = np.where(mrec[1::] != mrec[0:-1])[0] + 1
    ap = np.sum((mrec[idx] - mrec[idx - 1]) * mprec[idx])
    return ap


def cal_iou(probs, target, target_timings, gt_truth_timings, prop_scores):
    # target_timings = [[bs, 31], [bs, 51]]
    # ground_truth_timings = [bs, max_prop, 2]
    all_iou = []
    for video_index in range(gt_truth_timings.shape[0]):
        start_end_prop_index_list = []
        gt_start_end_list = []
        for prop_index in range(gt_truth_timings.shape[1]):
            start_timing_list = []
            end_timing_list = []
            for count, preds in enumerate(probs):
                start_index = preds[0][video_index, prop_index, :].argmax(0) - 1 # [bs, max_prop_start_index]
                end_index = preds[1][video_index, prop_index, :].argmax(0) - 1 # [bs, max_prop_start_index]
                prop_score = prop_scores[count][video_index, prop_index]
                start_index = start_index.item()
                end_index = end_index.item()
                if start_index == -1 or end_index == -1 or end_index < start_index or prop_score < threshold:
                    start_timing_list.append(0)
                    end_timing_list.append(0)
                    continue
                start_timing = target_timings[count][video_index, start_index].item()
                end_timing = target_timings[count][video_index, end_index + 1].item()
                start_timing_list.append(start_timing)
                end_timing_list.append(end_timing)


            
            start_end_prop_index_list.append(np.array([np.mean(start_timing_list), np.mean(end_timing_list)]))
            gt_start_end_list.append(np.array([gt_truth_timings[video_index, prop_index, 0], gt_truth_timings[video_index, prop_index, 1]]))

        start_end_prop_index_tensor = np.stack(start_end_prop_index_list)
        gt_start_end_tensor = np.stack(gt_start_end_list)

        gt_start_end_tensor = gt_start_end_tensor[gt_start_end_tensor[:, 0] < gt_start_end_tensor[:, 1]]
        start_end_prop_index_tensor = start_end_prop_index_tensor[start_end_prop_index_tensor[:, 0] < start_end_prop_index_tensor[:, 1]]
        start_end_prop_index_tensor = start_end_prop_index_tensor[:len(gt_start_end_tensor), :]
        gt_versus_pred = segment_iou(gt_start_end_tensor, start_end_prop_index_tensor)
        # gt_iou = np.max(gt_versus_pred, axis=1)
        # avg_iou.extend(gt_iou.tolist())
        all_iou.append(gt_versus_pred)

    return all_iou
 
def cal_recall(probs, target, target_timings, gt_truth_timings, prop_scores, iou_threshold):
    all_iou_list = cal_iou(probs, target, target_timings, gt_truth_timings, prop_scores)
    matches = np.zeros((gt_truth_timings.shape[0], iou_threshold.shape[0]))
    pos = np.zeros(gt_truth_timings.shape[0])
    # Matching
    recall = np.empty(iou_threshold.shape[0])
    for cidx, this_iou in enumerate(iou_threshold):
        # Score analysis per video.
        for i, sc in enumerate(all_iou_list):
            pos[i] = sc.shape[0] # Positives per video.
            lmt = int(sc.shape[1])
            matches[i, cidx] = ((sc[:, :lmt] >= this_iou).sum(axis=1) > 0).sum()
        this_recall = matches[:, cidx].sum() / pos.sum()
        recall[cidx] = this_recall
    return recall

def segment_iou(target_segments, test_segments):
    """Compute intersection over union btw segments
    Parameters
    ----------
    target_segments : ndarray
        2-dim array in format [m x 2:=[init, end]]
    test_segments : ndarray
        2-dim array in format [n x 2:=[init, end]]
    Outputs
    -------
    iou : ndarray
        2-dim array [m x n] with IOU ratio.
    Note: It assumes that target-segments are more scarce that test-segments
    """
    if target_segments.ndim != 2 or test_segments.ndim != 2:
        raise ValueError('Dimension of arguments is incorrect')

    m, n = target_segments.shape[0], test_segments.shape[0]
    iou = np.empty((m, n))
    for i in range(m):
        tt1 = np.maximum(target_segments[i, 0], test_segments[:, 0])
        tt2 = np.minimum(target_segments[i, 1], test_segments[:, 1])

        # Non-negative overlap score
        intersection = (tt2 - tt1).clip(0)
        union = ((test_segments[:, 1] - test_segments[:, 0]) +
                 (target_segments[i, 1] - target_segments[i, 0]) -
                 intersection)
        # Compute overlap as the ratio of the intersection
        # over union of two segments at the frame level.
        iou[i, :] = intersection / union
    return iou


# from NBT
def update_values(dict_from, dict_to):
    for key, value in dict_from.items():
        if isinstance(value, dict):
            update_values(dict_from[key], dict_to[key])
        elif value is not None:
            dict_to[key] = dict_from[key]


def calculate_bleu_meteor_scores(predicted_sentence, actual_sentence):
    score_map = {}

    gts = {}
    res = {}
    gts['0'] = []
    gts['0'].append({'caption': actual_sentence})
    res['0'] = []
    res['0'].append({'caption': predicted_sentence})

    tokenizer = PTBTokenizer()
    gts = tokenizer.tokenize(gts)
    res = tokenizer.tokenize(res)

    bleu_scores = Bleu(4).compute_score(gts, res)[1]
    meteor_scores = Meteor().compute_score(gts, res)[1]
    # score_map['bleu_1'][index] = score_map['bleu_1'][index] + bleu_scores[0]
    # score_map['bleu_1_count'][index] = score_map['bleu_1_count'][index] + 1
    # score_map['bleu_2'][index] = score_map['bleu_2'][index] + bleu_scores[1]
    # score_map['bleu_2_count'][index] = score_map['bleu_2_count'][index] + 1
    score_map['bleu_3'] = bleu_scores[2]
    score_map['bleu_4'] = bleu_scores[3]
    score_map['meteor'] = meteor_scores
    return score_map
