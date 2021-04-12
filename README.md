

# TREC

Please see `present.ipynb` for more details. If github is not showing the file properly, go to https://nbviewer.jupyter.org/github/philip-fu/TREC/blob/main/present.ipynb

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

## code structure
```
 ./
 │   README.md
 │   present.ipynb
 │   requrements.txt
 │   setup.sh
 │   main.py # script to train/eval models
 └───config/
 │   │   config.py # model parameters
 └───data/
 │   │   train_5500.label.txt
 │   │   test_TREC_10.label.txt
 └───model/
 │   │   nb.py # a baseline naive bayes model
 │   │   embedding_lstm.py # a lstm model
 └───src/
 │   │   data.py # functions to load data
 │   │   word_embedding.py # functions for word embedding and cleaning
 ```
