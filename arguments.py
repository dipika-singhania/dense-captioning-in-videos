import argparse
import os


def get_args():
    parser = argparse.ArgumentParser()
    base_dir = '/hpctmp/e0384157/captioning_data/'
    # Parameters not to changed generally
    parser.add_argument('--num_workers', type=int, default=4, help="Number of parallel thread to fetch the data")

    parser.add_argument('--features_base_dir', type=str, default=base_dir + '/video_features/',
                        help="Path to the data folder,  containing all datasets")
    parser.add_argument('--model_file_path', type=str,
                        default=base_dir + '/models_checkpoint/activity_net/',
                        help="Path to the directory where to save all models")
    parser.add_argument('--dataset_file', type=str,
                        default='data/anet_annotations_trainval.json',
                        help="File from where the annotations is picked up")
    parser.add_argument('--duration_file', type=str,
                        default='data/anet_duration_frame.csv',
                        help="File from where the frames per second of the video is picked up")
    parser.add_argument('--dataset', type=str, default='anet', choices=['anet', 'yc2'],
                        help="Dataset which we want to use")
    parser.add_argument('--out_path', type=str, default=base_dir + '/results/', help="")

    # Initial hyper-parameters to be fixed
    parser.add_argument('--lr', type=float, default=3e-4, help="Learning rate")
    parser.add_argument('--dec_lr', type=float, default=0.001, help="Decoder learning rate")
    parser.add_argument('--lr_end', type=float, default=3e-5, help="Learning rate")
    parser.add_argument('--batch_size', type=int, default=10, help="Batch Size")
    parser.add_argument('--epochs', type=int, default=50, help="Training epochs")

    parser.add_argument('--threshold', type=float, default=0.6, help='Threshold we take to filter scores fo VALIDATION')
    parser.add_argument('--sentence_train_start', type=float, default=0.4, help='When to start TRAINING of sentences')
    parser.add_argument('--iou_threshold', type=float, nargs='+',
                        default=[0.3, 0.5, 0.7, 0.9],
                        help="Threshold used for calculating recall scores")

    # Model specific parameters
    # parser.add_argument('--video_feat_dim', type=int, default=500, help='')  # 352 1024
    parser.add_argument('--feature_name', type=str, nargs='+', default=['_bn'])
    parser.add_argument('--feature_dim', type=int, default=3072, help='')
    parser.add_argument('--linear_dim', type=int, default=1024, help='')
    parser.add_argument('--embed_dim', type=int, default=1024, help='')

    #  Non local blocks parameters
    parser.add_argument('--latent_dim', type=int, default=512, help='')
    parser.add_argument('--dropout_rate', type=float, default=0.5, help='')
    parser.add_argument('--scale_factor', type=float, default=-.5, help='')
    parser.add_argument('--scale', type=bool, default=True, help='')

    # Event Proposal Module
    parser.add_argument('-p', '--dim_past_list', action='append', type=int,
                        help='Past seconds to be taken into account', required=False)
    parser.add_argument('-c', '--dim_cur_list', action='append', type=int,
                        help='Past seconds to be taken into account', required=False)
    parser.add_argument('--max_proposals', type=int, default=27, help="Max proposals to work with")

    # Captioning Proposal Module
    parser.add_argument('--max_sentence_len', type=int, default=20, help="Max sentence length ot decode")
    parser.add_argument('--teacher_forcing_ratio', type=float, default=0.6,
                        help="How much do we want the training to learn from actual gt")

    # Parameter defining what task to do, whether to pick up models etc.
    parser.add_argument('--pick_best', action='store_true', help="Pick the best model")
    parser.add_argument('--resume', action='store_true', help="Resumes from last trained model")
    parser.add_argument('--cal_scores', action='store_true', help="Whether we need to calculate bleu scores or not")
    parser.add_argument('--tensorboard', action='store_true', help="Whether we want to store the tensorboard results")
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'validate', 'test'],
                        help="Whether to perform training, validation or test. "
                             "If test is selected, --json_directory must be used to provide directory in which to save "
                             "the generated jsons.")
    parser.add_argument('--p_json', action='store_true', help="Do we want the json file to be formed")
    parser.add_argument('--p_result', action='store_true', help="Whether we want to print results in separate files")
    parser.add_argument('--log_E', type=str, default='out_1.log', help="Whether we want to log everything or not")
    parser.add_argument('--add_logit', action='store_true', help="Add logit loss")
    parser.add_argument('--no_tab', action='store_true', help="Do not add tab or lstm")
    parser.add_argument('--no_tab_no_lstm', action='store_true', help="Do not add tab or lstm")
    args = parser.parse_args()

    # Create folder according to dataset for saving models and json output
    if not os.path.exists(args.out_path):
        os.mkdir(args.out_path)

    if args.p_json:
        path = os.path.join(args.out_path, args.dataset, "json_logs")
        if not os.path.exists(path):
            os.mkdir(path)
        args.json_logs = path

    if args.tensorboard:
        path = os.path.join(args.out_path, args.dataset, "tensorboard")
        if not os.path.exists(path):
            os.mkdir(path)
        args.tb_logs = path

    text_logs = os.path.join(args.out_path, args.dataset, "text_logs")
    if not os.path.exists(text_logs):
        os.mkdir(text_logs)
    args.text_logs = text_logs

    args.add_logit = True
    return args
