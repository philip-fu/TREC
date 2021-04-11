#%%
import logging
import pickle
import fire

import numpy as np
from sklearn.metrics import confusion_matrix

from src.data import load
from model.nb import NBClassifier
from model.embedding_lstm import LSTMClassifier
from config.config import model_config

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# %%
def fit_baseline(X_train, y_train, X_test=None, y_test=None, save_to='model.pkl'):
    baseline = NBClassifier()
    baseline.fit(X_train, y_train)

    if X_test is not None:
        preds = baseline.infer(X_test)
        test_accuracy = round(np.mean(y_test == preds).item(), 2)
        logger.info("Accuracy is {}".format(test_accuracy))

    with open(save_to, 'wb') as output:
        pickle.dump(baseline, output, pickle.HIGHEST_PROTOCOL)

    logger.info("Model saved to {}".format(save_to))

    return baseline

def fit_model(X_train, y_train, X_test=None, y_test=None, save_to='model.pkl'):
    model = LSTMClassifier(model_config)
    model.fit(X_train, y_train)

    if X_test is not None:
        test_accuracy = model.evaluate(X_test, y_test)
        logger.info("Accuracy is {}".format(test_accuracy))

    with open(save_to, 'wb') as output:
        pickle.dump(model, output, pickle.HIGHEST_PROTOCOL)

    logger.info("Model saved to {}".format(save_to))

    return model

class Confusion:
    def __init__(self, cm, labels):
        self.cm = cm
        self.labels = labels

def eval_model(X_test, y_test, model):
    preds = model.infer(X_test)
    test_accuracy = round(np.mean(y_test == preds).item(), 2)

    labels = sorted(list(set(y_test)|set(preds)))
    cm = confusion_matrix(y_test, preds, labels=labels)
    cm = Confusion(cm=cm, labels=labels)

    return test_accuracy, cm

# %%
def train(train_data_path, eval_data_path=None, save_to='model.pkl', do_baseline=True):
    """Train and save models

    Args:
        train_data_path: str, path to data
        save_to: str, path where the model will be saved
        do_baseline: bool, whether to train baseline models.

    Returns,
        None.
    """
    train = load(train_data_path)
    if eval_data_path is not None:
        test = load(eval_data_path)
    else:
        test = {
            "question": None,
            "coarse_category": None,
            "fine_category": None
        }

    model_coarse = fit_model(X_train=train['question'], y_train=train['coarse_category'],
                             X_test=test['question'], y_test=test['coarse_category'],
                             save_to=save_to.split('.')[0] + '_coarse.pkl')

    model_finegr = fit_model(X_train=train['question'], y_train=train['fine_category'],
                             X_test=test['question'], y_test=test['fine_category'],
                             save_to=save_to.split('.')[0] + '_finegr.pkl')

    
    # baselines
    if do_baseline:
        baseline_coarse = fit_baseline(X_train=train['question'], y_train=train['coarse_category'],
                                       X_test=test['question'], y_test=test['coarse_category'],
                                       save_to=save_to.split('.')[0] + '_baseline_coarse.pkl')
        baseline_finegr = fit_baseline(X_train=train['question'], y_train=train['fine_category'],
                                       X_test=test['question'], y_test=test['fine_category'],
                                       save_to=save_to.split('.')[0] + '_baseline_finegr.pkl')

    

#%%
def evaluate(eval_data_path, 
             model_path_coarse,
             model_path_finegr,
             baseline_model_path_coarse,
             baseline_mdoel_path_finegr):
    """Evaluate the models with baselines.

    Args:
        eval_data_path: str, path to eval data.
        xxx_model_path_xxx: str, model paths

    Returns:
        cm_xxx: Confusion class.
    """
    with open(model_path_coarse, 'rb') as f:
        model_coarse = pickle.load(f)
    with open(model_path_finegr, 'rb') as f:
        model_finegr = pickle.load(f)
    
    test = load(eval_data_path)
    model_coarse_accuracy, cm_coarse = eval_model(X_test=test['question'], 
                                                  y_test=test['coarse_category'],
                                                  model=model_coarse)
    model_finegr_accuracy, cm_finegr = eval_model(X_test=test['question'], 
                                                  y_test=test['fine_category'],
                                                  model=model_finegr)
    
    # baselines
    with open(baseline_model_path_coarse, 'rb') as f:
        baseline_coarse = pickle.load(f)
    with open(baseline_mdoel_path_finegr, 'rb') as f:
        baseline_finegr = pickle.load(f)
    
    baseline_coarse_accuracy, _ = eval_model(X_test=test['question'], 
                                             y_test=test['coarse_category'],
                                             model=baseline_coarse)
    baseline_finegr_accuracy, _ = eval_model(X_test=test['question'], 
                                             y_test=test['fine_category'],
                                             model=baseline_finegr)
    
    # 
    logger.info("Coarse-grained accuracy {}% (vs {}% baseline)".format(model_coarse_accuracy, baseline_coarse_accuracy))
    logger.info("Fine-grained accuracy {}% (vs {}% baseline)".format(model_finegr_accuracy, baseline_finegr_accuracy))

    return cm_coarse, cm_finegr
    

    

    
if __name__ == "__main__":
    fire.Fire()
