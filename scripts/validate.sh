 python run_regression_encoding.py --mode validate --p_json -p 10 -p 15 -p 20 -c 50 --sentence_train_start 0.4 --add_logit
 python evaluations/evaluate.py --submission densecap_validation.json --out_file logit_50_remove_attn_results.csv

