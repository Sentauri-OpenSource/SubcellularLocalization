import os
import sys
import random
import shutil
import numpy as np 
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, f1_score, matthews_corrcoef
from sklearn.exceptions import UndefinedMetricWarning
import torch
from catboost import CatBoostClassifier
from joblib import dump, load
import optuna
import warnings 
import pdb

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="torch_geometric.utils.scatter")
warnings.simplefilter(action='ignore', category=FutureWarning)

dependant_variables = ['Golgi_Apparatus', 'Endoplasmic_Reticulum', 'Cytoplasm', 'Nucleus', 'Extracellular', 'Cell_Membrane', 'Mitochondria']

specific_label = dependant_variables[int(sys.argv[1])]
models_dir = sys.argv[2]

specific_dir = models_dir + specific_label.replace(' ', '') + '/'

try:
    os.mkdir(specific_dir)
except:
    pass

seed=42
def seed_torch():
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    return None

seed_torch()

class Logger(object):
    def __init__(self, filename="log.txt"):
        self.terminal = sys.stdout
        self.log_file = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

log = Logger()
sys.stdout = log

def train_model(parameter_dictionary):

    if parameter_dictionary['trial_num'] == 0:
        study_best = 0
        
    else:
        study_best = study.best_trial.values[0]

    total_train_preds, total_train_probs, total_train_labels = [], [], []
    total_val_preds, total_val_probs, total_val_labels = [], [], []

    num_folds =  5
    skf = StratifiedKFold(n_splits=num_folds)
    for i, (train_index, test_index) in enumerate(skf.split(training_concatenated_data, training_deeploc_labels)):

        train_data = training_concatenated_data.iloc[train_index]
        train_labels = training_deeploc_labels.iloc[train_index]

        val_data = training_concatenated_data.iloc[test_index]
        val_labels = training_deeploc_labels.iloc[test_index]

        total_train_labels += train_labels[specific_label].to_list()
        total_val_labels += val_labels[specific_label].to_list()

        model = CatBoostClassifier(task_type='GPU', grow_policy=parameter_dictionary['grow_policy'], iterations=parameter_dictionary['iterations'], learning_rate=parameter_dictionary['learning_rate'], depth=parameter_dictionary['depth'], l2_leaf_reg=parameter_dictionary['l2_leaf_reg'], bagging_temperature=parameter_dictionary['bagging_temperature'], random_strength=parameter_dictionary['random_strength'], scale_pos_weight=parameter_dictionary['scale_pos_weight'], loss_function='Logloss', border_count=128, random_seed=seed, verbose=False)
        model.fit(train_data, train_labels)

        train_preds = model.predict(train_data)
        val_preds = model.predict(val_data)

        total_train_preds += list(train_preds)
        total_val_preds += list(val_preds)

        train_probs = model.predict_proba(train_data)
        val_probs = model.predict_proba(val_data)

        total_train_probs += list(train_probs)
        total_val_probs += list(val_probs)

        fold_f1 = f1_score(val_labels, val_preds)
        print('Fold ' + str(i) + ' F1 : ' + str(fold_f1))

    cv_f1 = f1_score(total_val_labels, total_val_preds)
    print('F1 : ' + str(cv_f1))
    print()
    print(classification_report(total_train_labels, total_train_preds))
    print()
    print(classification_report(total_val_labels, total_val_preds))

    if cv_f1 > study_best:

        model = CatBoostClassifier(task_type='GPU', grow_policy=parameter_dictionary['grow_policy'], iterations=parameter_dictionary['iterations'], learning_rate=parameter_dictionary['learning_rate'], depth=parameter_dictionary['depth'], l2_leaf_reg=parameter_dictionary['l2_leaf_reg'], bagging_temperature=parameter_dictionary['bagging_temperature'], random_strength=parameter_dictionary['random_strength'], scale_pos_weight=parameter_dictionary['scale_pos_weight'], loss_function='Logloss', border_count=128, random_seed=seed, verbose=False)
        model.fit(training_concatenated_data, training_deeploc_labels)

        total_test_labels = testing_deeploc_labels[specific_label].to_list()

        total_test_preds = list(model.predict(testing_concatenated_data))
        total_test_probs = list(model.predict_proba(testing_concatenated_data))

        test_mcc = matthews_corrcoef(total_test_labels, total_test_preds)
        test_f1 = f1_score(total_test_labels, total_test_preds)
        print('Testing F1 : ' + str(test_f1))
        print('Testing MCC : ' + str(test_mcc))
        print()
        print(classification_report(total_test_labels, total_test_preds))

        test_probs_name = 'TestSetProbs_' + specific_label.replace(' ', '') + '.joblib'
        test_preds_name = 'TestSetPreds_' + specific_label.replace(' ', '') + '.joblib'
        test_labels_name = 'TestSetLabels_' + specific_label.replace(' ', '') + '.joblib'

        val_probs_name = 'ValSetProbs_' + specific_label.replace(' ', '') + '.joblib'
        val_preds_name = 'ValSetPreds_' + specific_label.replace(' ', '') + '.joblib'
        val_labels_name = 'ValSetLabels_' + specific_label.replace(' ', '') + '.joblib'

        train_probs_name = 'TrainSetProbs_' + specific_label.replace(' ', '') + '.joblib'
        train_preds_name = 'TrainSetPreds_' + specific_label.replace(' ', '') + '.joblib'
        train_labels_name = 'TrainSetLabels_' + specific_label.replace(' ', '') + '.joblib'

        dump(total_test_probs, test_probs_name)
        dump(total_test_preds, test_preds_name)
        dump(total_test_labels, test_labels_name)

        dump(total_val_probs, val_probs_name)
        dump(total_val_preds, val_preds_name)
        dump(total_val_labels, val_labels_name)

        dump(total_train_probs, train_probs_name)
        dump(total_train_preds, train_preds_name)
        dump(total_train_labels, train_labels_name)

        model_name = 'CatBoostModel_' + specific_label.replace(' ', '') + '.cbm'
        model.save_model(model_name)

        shutil_dir = specific_dir + 'Trial_' + str(parameter_dictionary['trial_num']) + '/'
        shutil_list = [val_probs_name, val_preds_name, val_labels_name, train_probs_name, train_preds_name, train_labels_name, test_probs_name, test_preds_name, test_labels_name, model_name]
        try:
            os.mkdir(shutil_dir)
        except:
            pass

        try:
            for i in shutil_list:
                shutil.move(i, shutil_dir)

        except:
            pass

    return cv_f1

def objective(trial):

    seed_torch()
    trial_num = trial.number

    parameters = {
        
            'grow_policy' : trial.suggest_categorical('grow_policy', ['SymmetricTree', 'Depthwise']),
            'iterations' : trial.suggest_int('iterations', 100, 2000, step=50),
            'learning_rate' : trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
            'depth' : trial.suggest_int('depth', 3, 6),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 0.1, 10, log=True),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
            'random_strength': trial.suggest_float('random_strength', 0.1, 1.0),
            'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1.0, 10.0),

            'trial_num' : trial_num,
            }

    print('\n')
    print(parameters)
    print()

    trial_f1 = train_model(parameters)

    return trial_f1

#############################################################################################################################################################################

triplet_embedding_dir = '/'.join(models_dir.split('/')[0:-3]) + '/'
training_triplet_embedding_data = load(triplet_embedding_dir + 'Sentauri_DeepLocTraining_Triplet_Embeddings.joblib')

protbert_embedding_dir = '/'.join(models_dir.split('/')[0:-5]) + '/'
training_protbert_embedding_data = load(protbert_embedding_dir + 'Sentauri_DeepLoc_ProtBERT_Embeddings_Training.joblib')
validation_protbert_embedding_data = load(protbert_embedding_dir + 'Sentauri_DeepLoc_ProtBERT_Embeddings_Validation.joblib')
training_protbert_embedding_data = pd.concat([training_protbert_embedding_data, validation_protbert_embedding_data], axis=0)

training_protbert_embedding_data = training_protbert_embedding_data.drop(dependant_variables, axis=1)
training_protbert_embedding_data = training_protbert_embedding_data.loc[training_triplet_embedding_data.index.to_list()]

training_concatenated_data = pd.concat([training_protbert_embedding_data, training_triplet_embedding_data], axis=1)
training_deeploc_labels = training_concatenated_data[[specific_label]]

training_concatenated_data = training_concatenated_data.drop(dependant_variables, axis=1)
training_concatenated_data.columns = [i for i in range(len(training_concatenated_data.columns.to_list()))]

training_accessions = training_concatenated_data.index.to_list()

clusters = load(protbert_embedding_dir + 'Optimal_DBSCAN_Clusters.joblib')
original_dir = '/'.join(models_dir.split('/')[0:-6]) + '/'
original_df = load(original_dir + 'Sentauri_DeepLoc_ProtBERT_Embeddings.joblib')
original_df['Cluster'] = clusters
original_df = original_df[['Cluster']]
cluster_dict = original_df.to_dict()['Cluster']

training_groups = [cluster_dict[i] for i in training_accessions]

testing_triplet_embedding_data = load(triplet_embedding_dir + 'Sentauri_DeepLocTesting_Triplet_Embeddings.joblib')
testing_protbert_embedding_data = load(protbert_embedding_dir + 'Sentauri_DeepLoc_ProtBERT_Embeddings_Testing.joblib')
testing_protbert_embedding_data = testing_protbert_embedding_data.drop(dependant_variables, axis=1)

testing_concatenated_data = pd.concat([testing_protbert_embedding_data, testing_triplet_embedding_data], axis=1)

testing_deeploc_labels = testing_concatenated_data[[specific_label]]
testing_concatenated_data = testing_concatenated_data.drop(dependant_variables, axis=1)
testing_concatenated_data.columns = [i for i in range(len(testing_concatenated_data.columns.to_list()))]

testing_accessions = testing_protbert_embedding_data.index.to_list()

study = optuna.create_study(direction='maximize',sampler=optuna.samplers.TPESampler(seed=seed))
study.optimize(objective, n_trials=100, n_jobs=1, show_progress_bar=True)
best_params = study.best_trial.params

print(best_params)

dump(best_params, 'BestCatBoostHyperparams_Triplet_Embeddings_' + specific_label.replace(' ', '') + '.joblib')
study.trials_dataframe().to_csv('OptunaData_' + specific_label.replace(' ', '') + '.csv', index=False)

dump(study, 'OptunaStudy_' + specific_label.replace(' ', '') + '.joblib')
