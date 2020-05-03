# Packages which needs to be install to run the codes
pip install tensorboardX
pip install git+https://github.com/salaniz/pycocoevalcap
python -m spacy download en (pip --user)


# pycocoevalcap requires java in your machine


Some of the packages are mentioned in requirements.txt but some of them needs to be run manually


1. Setting up the environment.

Some of the packages are mentioned in requirements.txt but some of them needs to be run manually

pip install -r requirements.txt
pip install tensorboardX
pip install git+https://github.com/salaniz/pycocoevalcap
python -m spacy download en (pip --user)
# pycocoevalcap requires java in your machine

2. Data preparation and output directory path confirguration

Data needs to be downloaded from https://github.com/LuoweiZhou/densecap captioning paper.
One can even download other features and update the arguments

--features_base_dir : Base directory of the features
--out_path : output has to be updated

Inside the output path: json_logs, tensorboard and text_logs is formed

3. Training scripts: some training scripts are updated in scripts/ folder.

Keep the default hyperparameters, one can run the following command to train the model

CUDA_VISIBLE_DEVICES=6 python run_regression_encoding.py --mode train --tensorboard

4. validation scripts: keeping the default paramters one can pick up the last saved model
and run the validation through the following command

# CUDA_VISIBLE_DEVICES=6 python run_regression_encoding.py --mode validate --p_json
python run_regression_encoding.py --mode validate --p_json -p 15 -p 20  -c 30 -c 50 --sentence_train_start 0.4 --add_logit

--p_json arguments forms the json file which is stored in a directory args.out_path/anet/json_logs/model_name_formed/densecap_validation.json

This json file need to be send to the evaluation/evaluate.py script to get the final evaluations scores.

python evaluations/evaluate.py --submission  args.out_path/anet/json_logs/model_name_formed/densecap_validation.json


5. Tensorboard logs: If --tensorboard option is specified, tensorboard logs are dumped in directory args.out_path/anet/tensorboard/model_name_formed/model_name_formed/

To visualize the tensorboard results one needs to run the following command:

tensorboard --logdir args.out_path/anet/tensorboard/model_name_formed/model_name_formed/ --port 9900

Then one can visualize the logs through the browser at localhost:9900

Model checkpoint is available 
