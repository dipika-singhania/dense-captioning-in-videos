from torch import nn
from .non_local_embedded_gaussian import NONLocalBlock1D
import torch
import torch.nn.functional as F
import random


class Network_sunNON(nn.Module):
    def __init__(self, non_local_params, dim_curr, dim_past, linear_dim):
        super(Network_sunNON, self).__init__()

        self.convBlock1_all = NONLocalBlock1D(non_local_params, dim_past, dim_past, linear_dim)
        self.convBlock2_all = NONLocalBlock1D(non_local_params, dim_curr, dim_past, linear_dim)
        # layers = 4
        # self.convBlock1_list = nn.ModuleList([NONLocalBlock1D(non_local_params, dim_past, dim_past, linear_dim) for i in range(layers)])
        # self.convBlock2_list = nn.ModuleList([NONLocalBlock1D(non_local_params, dim_curr, dim_past, linear_dim) for i in range(layers)])

    def forward(self, x_past_actual, x_curr_actual):
        # x_curr_actual.shape = torch.Size([64, 5, 400])
        # x_past_actual.shape = torch.Size([64, 25, 400])
        batch_size = x_past_actual.size(0)  # batch_size = 64
        nle_x_past = F.relu(self.convBlock1_all(x_past_actual, x_past_actual))
        nle_x_future = F.relu(self.convBlock2_all(nle_x_past, x_curr_actual))  # output.shape = torch.Size([64, 5, 400])

        # for m1, m2 in zip(self.convBlock1_list, self.convBlock2_list):
        #     nle_x_past = F.relu(m1(nle_x_past, nle_x_past))
        #     nle_x_future = F.relu(m2(nle_x_past, nle_x_future))
        return nle_x_future, nle_x_past


# Attention with all different past for a single recent
class NetworkinNetwork(nn.Module):
    def __init__(self, non_local_params, dim_past_list, dim_curr, linear_dim):
        super(NetworkinNetwork, self).__init__()

        self.convNONfc_list = nn.ModuleList(
            [Network_sunNON(non_local_params, dim_curr, dim_past, linear_dim) for dim_past in dim_past_list])

        self.lin_concat_future = nn.Sequential(
            nn.Linear(in_features=len(self.convNONfc_list) * linear_dim, out_features=linear_dim),
            nn.LayerNorm(torch.Size([dim_curr, linear_dim])),
            nn.ReLU(),
            nn.Dropout(p=non_local_params['dropout_rate'])
        )

    #         self.lin_concat_past = nn.Sequential(
    #             nn.Linear(in_features = linear_dim, out_features=linear_dim),
    #             nn.LayerNorm(torch.Size([linear_dim, linear_dim])),
    #             nn.ReLU(),
    #             nn.Dropout(p=non_local_params['dropout_rate'])
    #         )

    def forward(self, x_past_actual_all, x_curr_actual):
        netFuture_list = []
        net_past_list = []
        for i, convNonfc in enumerate(self.convNONfc_list):
            netFuture_s1, nle_past = convNonfc(x_past_actual_all[i], x_curr_actual)
            netFuture_list.append(netFuture_s1)
        #             net_past_list.append(nle_past)

        comb_netFuture = torch.cat(netFuture_list, 2)
        comb_netFuture = self.lin_concat_future(comb_netFuture)

        #         comb_net_past = torch.cat(net_past_list, 1)
        #         comb_net_past = torch.max(comb_net_past, 1)[0]
        #         comb_net_past = self.lin_concat_past(comb_net_past)
        return comb_netFuture


# Ensemble of different recents
class Tab(nn.Module):
    def __init__(self, non_local_params, dim_past_list, curr_sec_list, linear_dim):
        super(Tab, self).__init__()

        self.NetInNet_list = nn.ModuleList(
            [NetworkinNetwork(non_local_params, dim_past_list, dim_cur, linear_dim) for dim_cur in curr_sec_list])

        self.cls_future_list = nn.ModuleList([nn.Sequential(nn.Linear(in_features=linear_dim, out_features=linear_dim),
                                                            nn.LayerNorm(torch.Size([dim_curr, linear_dim])),
                                                            nn.ReLU(),
                                                            nn.Dropout(p=non_local_params['dropout_rate'])
                                                            ) \
                                              for dim_curr in curr_sec_list])

    def forward(self, x_past_actual_all, x_curr_actual_all):
        output_final_future_list = []
        for i, NetInNet in enumerate(self.NetInNet_list):
            comb_netFuture = NetInNet(x_past_actual_all, x_curr_actual_all[i])
            #             comb_netFuture_netPast = torch.cat((comb_netFuture, comb_netPast), 1)  # output_future_task_fc.shape  = torch.Size([64, 2048])
            output_final_future = self.cls_future_list[i](
                comb_netFuture)  # output_final_future.shape    = torch.Size([64, 48])
            output_final_future_list.append(output_final_future)
        return output_final_future_list


# LSTM outputs for the proposals
class EncodingProposal(nn.Module):
    def __init__(self, non_local_params, encoding_proposal_params, enc_dim, feature_dim):
        super(EncodingProposal, self).__init__()
        self.proposals = encoding_proposal_params['max_proposals']
        self.embed_layer = nn.Linear(in_features=feature_dim, out_features=enc_dim)
        self.dropout = nn.Dropout(non_local_params['dropout_rate'])
        self.tab_layer = Tab(non_local_params, encoding_proposal_params['dim_past_list'],
                             encoding_proposal_params['dim_cur_list'], \
                             enc_dim)
        self.lstm_layer_list = nn.ModuleList([nn.LSTM(input_size=enc_dim, hidden_size=enc_dim, batch_first=True, \
                                                      bidirectional=True) \
                                              for _ in encoding_proposal_params['dim_cur_list']])
        self.lstm_non_linear = nn.ModuleList(
            [nn.Linear(enc_dim * 2, enc_dim) for _ in encoding_proposal_params['dim_cur_list']])

        dec_hid_dim = enc_dim
        self.enc_dim = enc_dim
        self.dec_hid_dim = enc_dim
        self.attn_list = nn.ModuleList([nn.Linear((enc_dim * 2) + enc_dim, dec_hid_dim) \
                                        for _ in encoding_proposal_params['dim_cur_list']])

        self.v_list = nn.ModuleList([nn.Linear(dec_hid_dim, self.proposals, bias=False) \
                                     for _ in encoding_proposal_params['dim_cur_list']])
        self.encode_final = nn.ModuleList([nn.Sequential(nn.LayerNorm(torch.Size([self.proposals, dec_hid_dim])),
                                                         nn.ReLU(),
                                                         nn.Linear(dec_hid_dim, dec_hid_dim)
                                                        ) for _ in
                                           encoding_proposal_params['dim_cur_list']])

        self.concat_and_linear = nn.ModuleList([nn.Sequential(nn.Linear(dec_hid_dim + dim_curr, dec_hid_dim),
                                                              nn.LayerNorm(torch.Size([self.proposals, dec_hid_dim])),
                                                              nn.ReLU(),
                                                              nn.Dropout(p=non_local_params['dropout_rate'])) for
                                                dim_curr in encoding_proposal_params['dim_cur_list']])

        # self.proposal_score = nn.ModuleList([nn.Sequential(nn.Linear(dec_hid_dim, 3),
        #                                                    nn.Sigmoid()) for dim_curr in
        #                                      encoding_proposal_params['dim_cur_list']])
        self.proposal_score = nn.ModuleList([nn.Sequential(nn.Linear(dec_hid_dim, 2),
                                                           nn.Sigmoid()) for _ in
                                             encoding_proposal_params['dim_cur_list']])
        self.score_sentence = nn.ModuleList([nn.Linear(dec_hid_dim, 1) for _ in encoding_proposal_params['dim_cur_list']])

    def forward(self, x_past_actual_all, x_curr_actual_all):
        embed_past = []
        for ele_past in x_past_actual_all:
            embed_past.append(self.dropout(self.embed_layer(ele_past)))
        embed_recent = []
        for ele_recent in x_curr_actual_all:
            embed_recent.append(self.dropout(self.embed_layer(ele_recent)))
        tab_output_list = self.tab_layer(embed_past, embed_recent)

        encoded_features_list = []
        hidden_output_list = []
        proposals_score_list = []
        sentence_score_list = []
        for i, lstm_layer in enumerate(self.lstm_layer_list):
            seq_len = tab_output_list[i].shape[1]  # seq len = recent[0], 30
            output_lstm, (hidden, c_n) = lstm_layer(tab_output_list[i])  
            # output_lstm = [4, 30, 512], hidden = [2, 4, 256]
            # output_lstm, (hidden, c_n) = lstm_layer(embed_recent[i])
            encoded_features_list.append(output_lstm)

            hidden = torch.tanh(self.lstm_non_linear[i](torch.cat((hidden[-2, :, :],
                                                                   hidden[-1, :, :]), dim=1)))  # hidden = [4, 256]
            hidden_output_list.append(hidden)

            # repeat decoder hidden state seq_len times
            hidden = hidden.unsqueeze(1).repeat(1, seq_len, 1)
            # hidden = (Batch Size  * Seq Len * dec_hid_dim) = [4, 30, 256]

            encoding = self.attn_list[i](torch.cat((hidden, output_lstm), dim=2))
            # encoding = Batch_size * seq len * dec_hid_dim = [4, 30, 256]

            proposals = torch.softmax(self.v_list[i](encoding), dim=-1)  # Batch_size * seq len * max_proposals [4, 30, 27]

            proposals = proposals.permute(0, 2, 1)  # proposals = [4, 27, 31]

            curr_start_enc = torch.matmul(proposals, encoding)
            curr_start_transformed = self.encode_final[i](curr_start_enc)
            curr_start_enc = curr_start_enc + curr_start_transformed
            # curr_start_enc = (Batch_size * max_proosals * embed_size) = [4, 27, 256]

            # Check whether this is needed or not ---- 1
            curr_start_enc_with_proposals = self.concat_and_linear[i](torch.cat((curr_start_enc, proposals), dim=2))

            proposals_score_list.append(self.proposal_score[i](curr_start_enc_with_proposals))
            sentence_score_list.append(self.score_sentence[i](curr_start_enc_with_proposals))

        # encoded_features = torch.mean(torch.stack(encoded_features_list, dim=2), dim=2)
        # hidden_features = torch.mean(torch.stack(hidden_output_list, dim=2), dim=2)
        mean_proposal_score_1 = torch.mean(torch.stack(proposals_score_list, dim=2), dim=2)
        mean_proposal_score_2 = torch.mean(torch.stack(sentence_score_list, dim=2), dim=2)
        mean_proposal_score = torch.cat([mean_proposal_score_1, mean_proposal_score_2], dim=-1)
        return mean_proposal_score, encoded_features_list, hidden_output_list


class CaptioningEncoderProposal(nn.Module):
    def __init__(self, max_proposals, linear_dim, encoding_proposal_params):
        super(CaptioningEncoderProposal, self).__init__()
        self.enc_dim = linear_dim
        self.dec_hid_dim = linear_dim
        self.max_proposal = max_proposals

        # Captioning Encoder with Attention
        self.attn_list = nn.ModuleList([nn.Linear(self.enc_dim * 2, self.max_proposal, bias=False) \
                                        for _ in encoding_proposal_params['dim_cur_list']])
        self.encode_final_list = nn.ModuleList([nn.Sequential(\
                                                nn.LayerNorm(torch.Size([self.max_proposal, self.enc_dim * 2])),
                                                nn.ReLU(),
                                                nn.Linear(self.enc_dim * 2, self.enc_dim * 2),
                                               ) for _ in encoding_proposal_params['dim_cur_list']])

        len_recent = len(encoding_proposal_params['dim_cur_list'])
        self.v = nn.Linear(self.enc_dim * 2 * len_recent, self.max_proposal, bias=False)
        self.v_linear = nn.Sequential(nn.LayerNorm(torch.Size([self.max_proposal, self.enc_dim * 2 * len_recent])),
                                      nn.ReLU(),
                                      nn.Linear(self.enc_dim * 2 * len_recent, self.enc_dim * 2 * len_recent))

    def forward(self, encoder_outputs, hidden, mask):
        # hidden = [batch size, enc hid dim]
        # encoder_outputs = [batch size, max_proposal, enc hid dim * 2]
        # mask = [batch size, max_proposal]

        max_prop_len = mask.shape[1]
        count = 0
        encoded_proposal_list = []
        attn_weights_list = []
        for attn, encode, encode_out in zip(self.attn_list, self.encode_final_list, encoder_outputs):
            attn_weigths = torch.sigmoid(attn(encoder_outputs[count]))    # encoder_outputs = [bs, 30, 512],
            # attn_weights = [bs, 30, 27]
            attn_weigths = attn_weigths.permute(0, 2, 1)   # attn_weights = [bs, 27, 30]
            mask_attn = mask.unsqueeze(2).repeat(1, 1, attn_weigths.shape[2])
            attn_weigths.masked_fill(mask_attn == 0, 0)
            attn_weights_list.append(attn_weigths)
            encoded_proposal = torch.matmul(attn_weigths, encode_out)
            encoded_proposal_linear = encode(encoded_proposal)
            encoded_proposal = encoded_proposal + encoded_proposal_linear  # Does a residual of linear connection
            encoded_proposal_list.append(encoded_proposal)
            count = count + 1

        energy = torch.cat(encoded_proposal_list, dim=-1)
        return energy, attn_weights_list

        # Check whether this is required ---- 2
        # attention_u = self.v(energy)  # [batch_size, max_proposal, max_proposal]
        # # repeat mask for max_proposal times
        # mask_u = mask.unsqueeze(2).repeat(1, 1, max_prop_len)  # [Batch Size , max_proposal, max_proposal]
        # attention_u = attention_u.masked_fill(mask_u == 0, -1e10)
        # # attention_u = [batch Size, max_proposal, max_proposal] = [4, 27, 27]
        # a = F.softmax(attention_u, dim=1)  # a = [Batch Size, max_proposal, max_proposal] = [4, 27, 27]
        # weighted = torch.bmm(a, energy)
        # weighted = weighted + energy     # Add residual after attention
        # # weighted = [batch size, max_proposal, enc hid dim * 2]
        # weighted_ff = self.v_linear(weighted)
        # weighted = weighted + weighted_ff

        # return weighted, attn_weights_list


class CaptioningDecoderProposal(nn.Module):
    def __init__(self, max_proposals, linear_dim, embed_dim, vocab_size, pad_idx, len_recent):
        super(CaptioningDecoderProposal, self).__init__()
        self.enc_dim = linear_dim
        self.dec_hid_dim = linear_dim
        self.max_proposal = max_proposals

        # Captioning module
        self.embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim, padding_idx=pad_idx)
        self.dropout = nn.Dropout()
        self.rnn = nn.LSTM((self.enc_dim * 2 * len_recent) + embed_dim, (self.enc_dim * 2 * len_recent), batch_first=True)
        self.fc = nn.Linear((self.enc_dim * 2 * len_recent) + (self.enc_dim * 2 * len_recent) + embed_dim, vocab_size)

    def forward(self, input_sentences, filtered_proposals, rnn_hidden_prev, rnn_c_prev):
        #####################Captioning Decoder#################################    

        # input_sentences = [all_proposals_batch]
        embedded = self.dropout(self.embed(input_sentences))
        # embedded = [all_proposals_batch, emb dim]

        rnn_input = torch.cat((embedded.unsqueeze(1), filtered_proposals.unsqueeze(1)), dim=2)
        # rnn_input = [batch size, (enc hid dim * 4) + emb dim] = [10, 1, 768]

        output_cap, (rnn_hidden_curr, rnn_c_curr) = self.rnn(rnn_input,
                                                             (rnn_hidden_prev.unsqueeze(0), rnn_c_prev.unsqueeze(0)))

        # output_cap = [all_proposals_batch, 1, dec_hid_dim] = [10, 1, 512]
        # rnn_hidden_curr = [ 1, all_proposals_batch, dec hid dim] = [1, 10, 512]

        assert (output_cap.squeeze(1) == rnn_hidden_curr.squeeze(0)).all()

        embedded = embedded.squeeze(1)
        output_cap = output_cap.squeeze(1)
        filtered_proposals = filtered_proposals.squeeze(1)

        prediction = self.fc(torch.cat((output_cap, filtered_proposals, embedded), dim=1))

        return prediction, rnn_hidden_curr.squeeze(0), rnn_c_curr.squeeze(0)


def cal_iou_one_dim(p_gt_strt, p_gt_end, p_pred_strt, p_pred_len):
    p_gt_strt = p_gt_strt.squeeze()
    p_gt_end = p_gt_end.squeeze()
    p_pred_strt = p_pred_strt.squeeze()
    p_pred_len = p_pred_len.squeeze()

    p_pred_end = p_pred_strt + p_pred_len
    intersection_start = torch.max(p_gt_strt, p_pred_strt)
    intersection_end = torch.min(p_gt_end, p_pred_end)
    intersection = torch.clamp(intersection_end - intersection_start, 0)

    union_start = torch.min(p_gt_strt, p_pred_strt)
    union_end = torch.max(p_gt_end, p_pred_end)
    union = torch.clamp(union_end - union_start, 1)

    iou_scores = torch.div(intersection, union)

    return iou_scores


class OverallNetwork(nn.Module):
    def __init__(self, non_local_params, event_proposal_params, encoding_proposal_params, captioning_proposal_params,
                 feature_dim, linear_dim, embed_dim, vocab_size, pad_idx, iou_threshold, vocab):
        super(OverallNetwork, self).__init__()
        self.vocab = vocab
        self.max_proposals = event_proposal_params['max_proposals']
        self.threshold = event_proposal_params['threshold']
        self.iou_threshold = iou_threshold
        self.vocab_size = vocab_size
        print("Captioning proposal params:", captioning_proposal_params)
        self.max_sent_len = captioning_proposal_params['max_sentence_len']
        self.event_proposal = EncodingProposal(non_local_params, event_proposal_params, linear_dim, feature_dim)
        self.encode_proposal = CaptioningEncoderProposal(self.max_proposals, linear_dim, event_proposal_params)
        self.decode_proposal = CaptioningDecoderProposal(self.max_proposals, linear_dim, embed_dim, vocab_size, pad_idx,
                                                         len(event_proposal_params['dim_cur_list']))

    def forward(self, input_sentences, x_past_actual_all, x_curr_actual_all, gt_truth_timings, is_train,
                teacher_forcing_ratio, duration):
        # gt_truth_timings = [bs, max_prop, 2] 
        # x_past_actual_all = [[bs, 10, feature_dim], [bs, 15, feature_dim], [bs, 20, feature_dim]]
        # x_cur_actual_all = [[bs, 30, feature_dim], [bs, 50, feature_dim]]
        proposal_score, encoded_proposals, hidden_vector = self.event_proposal(x_past_actual_all, x_curr_actual_all)
        # proposal_score = [[bs, max_proposals, 3], 1st start, 2nd is length, 3rd is Score
        # encoded_proposals = [bs, max_proposals, enc_dim] = [4, 27, 256]
        # hidden_vector = [bs, 2 * enc_dim] = [4, 512]

        # Calculate the mask for valid proposals only, for train valid proposals is length > 0,
        # for validation valid proposals is score > threshold
        if is_train:
            gt_truth_timings_start = gt_truth_timings[:, :, 0].squeeze()  # gt_truth_timings_start = [bs, max_proposals
            gt_truth_timings_end = gt_truth_timings[:, :, 1].squeeze()  # gt_truth_timings_end = [bs, max_proposals]
            gt_truth_timings_length = gt_truth_timings_end - gt_truth_timings_start
            two_dim_mask = gt_truth_timings_length > 0

            # Calculate proposals start and end and find iou greater than iou threshold
            start_score = proposal_score[:, :, 0].squeeze()
            duration_expanded = duration.unsqueeze(1).expand_as(start_score)
            predicted_start_proposal = start_score * duration_expanded
            predicted_start_proposal_flt = predicted_start_proposal.squeeze().contiguous().view(-1, 1)

            length_score = proposal_score[:, :, 1].squeeze()
            predicted_len_proposal = (duration_expanded - predicted_start_proposal) * length_score
            predicted_len_proposal_flt = predicted_len_proposal.view(-1, 1)

            iou_scores = cal_iou_one_dim(gt_truth_timings_start.view(-1, 1), gt_truth_timings_end.view(-1, 1),
                                         predicted_start_proposal_flt, predicted_len_proposal_flt)
            filter_m = iou_scores > self.iou_threshold
        else:
            val_score = torch.sigmoid(proposal_score[:, :, 2].squeeze())  # val_score = [bs, max_proposals]
            two_dim_mask = val_score > self.threshold
            filter_m = two_dim_mask.view(-1, 1).squeeze(1)

        weighted_enc_props, attn_weights_list = self.encode_proposal(encoded_proposals, hidden_vector, two_dim_mask)

        if torch.sum(filter_m) < 2:
            return proposal_score, None, None, attn_weights_list

        # weighted_enc_props = [bs, max_proposals, 2 * linear_dim] = [4, 27, 512]
        flattened_enc_proposals = weighted_enc_props.view((-1, 1, weighted_enc_props.shape[2])).squeeze(1)
        # flattened_enc_proposals = [108, 512]
        filtered_encoded_proposal = flattened_enc_proposals[filter_m]
        # filtered_proposals = [all_proposals_batch, 2*self.enc_dim]

        if input_sentences is None and teacher_forcing_ratio is True:
            assert False

        target_sentences = None
        if is_train is False:
            filtered_sentences = torch.ones((filtered_encoded_proposal.shape[0]), dtype=torch.long,
                                            device=filtered_encoded_proposal.device) * self.vocab.vocab.stoi['<init>']

        else:
            # input_sentences = [bs, max_porposals, max_sentences_len]
            all_sentences = input_sentences.view((-1, 1, self.max_sent_len)).squeeze(1)
            target_sentences = all_sentences[filter_m]
            # target_sentences = [all_proposals_batch, max_sentence_len] = [108, 20]
            filtered_sentences = target_sentences[:, 0]  # filtered_sentences = [all_proposals_batch]

        assert (filtered_sentences.shape[0] == filtered_encoded_proposal.shape[0])
        rnn_hidden_prev = filtered_encoded_proposal
        rnn_c_prev = filtered_encoded_proposal

        # tensor to store decoder outputs
        outputs = torch.zeros(filtered_sentences.shape[0], self.max_sent_len, self.vocab_size).to(
            filtered_sentences.device)

        outputs[:, 0, self.vocab.vocab.stoi['<init>']] = 1

        # Call captioning decoder proposal in loop
        for t in range(self.max_sent_len - 1):
            prediction, rnn_hidden_curr, rnn_c_curr = \
            self.decode_proposal(filtered_sentences, filtered_encoded_proposal, rnn_hidden_prev, rnn_c_prev)
            # prediction = [10, 4563], rnn_hidden_curr=[10, 512], rnn_c_curr=[10,512]

            outputs[:, t + 1, :] = prediction  # prediction = [all_proposals_batch, vocab_size]

            # decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio

            # get the highest predicted token from our predictions
            top1 = prediction.argmax(1)  # [all_proposals_batch]

            # if teacher forcing, use actual next token as next input
            # if not, use predicted token
            filtered_sentences = target_sentences[:, t + 1] if is_train and teacher_force else top1

        # splitted_sentences =
        # torch.split(outputs, split_size_or_sections=count_of_number_of_sentences_per_video, dim=0)
        return proposal_score, outputs, filter_m, attn_weights_list
