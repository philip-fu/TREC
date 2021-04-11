## setup
```
bash setup.sh
```

## train
```
python main.py train --train_data_path=<path_to_train_txt> --eval_data_path=<path_to_test_txt> --save_to=<path_to_model_output>
```

## eval
```
python main.py evaluate --eval_data_path=<path_to_test_txt> \
--model_path_coarse=<path_to_coarse_model> \
--model_path_coarse=<path_to_finegrained_model> \
--model_path_coarse=<path_to_coarse_baseline_model> \
--model_path_coarse=<path_to_finegrained_baseline_model>
```
## env

Some parameters can be passed via env variable. See `config/config.py` for more details.


# TREC
