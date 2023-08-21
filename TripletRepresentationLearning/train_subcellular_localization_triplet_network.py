import os
import sys
import shutil
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import UndefinedMetricWarning
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from joblib import dump, load
import optuna
import warnings 
from utils import seed_torch, worker_init_fn
from models import TripletEmbeddingDataset, TripletNet
import pdb

'''Globals'''
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="torch_geometric.utils.scatter")
torch.set_float32_matmul_precision('high')

cwd = os.getcwd() + '/'
trials_dir = cwd + 'Trials/'
try:
    os.mkdir(trials_dir)
except:
    pass

seed=42
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

seed_torch()

def train_model(parameter_dictionary):

    trial_num = parameter_dictionary['trial_num']

    if trial_num < number_jobs:
        study_best = 10000
        
    else:
        study_best = study.best_trial.values[0]
    
    current_val_loss = 10000
    outmodel_name = 'DeepLoc_TripletNetwork_' + str(trial_num) + '.pt'

    train_data = TripletEmbeddingDataset(training_triplets, training_df_scaled, device)
    val_data = TripletEmbeddingDataset(validation_triplets, validation_df_scaled, device)
    test_data = TripletEmbeddingDataset(testing_triplets, testing_df_scaled, device)

    training_dataloader = DataLoader(train_data, batch_size=parameter_dictionary['batch_size'], shuffle=True, worker_init_fn=worker_init_fn)
    validation_dataloader = DataLoader(val_data, batch_size=parameter_dictionary['batch_size'], shuffle=False, worker_init_fn=worker_init_fn)
    testing_dataloader = DataLoader(test_data, batch_size=parameter_dictionary['batch_size'], shuffle=False, worker_init_fn=worker_init_fn)

    triplet_net = TripletNet(parameter_dictionary).to(device)

    optimizer = optim.Adam(triplet_net.parameters(), lr=parameter_dictionary['initial_learning_rate'])

    train_steps_for_val = np.linspace(0, len(training_dataloader) - 1, 10, dtype=int)
    scheduler = StepLR(optimizer, step_size=1, gamma=parameter_dictionary['gamma'])

    patience_counter = 0
    max_patience_counter = 2
    patience_exceeded = False

    train_loss_list = []
    val_loss_list = []
    
    for epoch in range(1, parameter_dictionary['epochs'] + 1):

        triplet_net.train()

        train_loss_accumulator = 0
        
        for train_step, training_data_batch in enumerate(training_dataloader):

            if not patience_exceeded:

                if train_step != 0:
                    print('Trial Num : ' + str(trial_num) + ' Epoch: ' + str(epoch) + ' Step: ' + str(train_step) + ' Training Loss: ' + "{:.8f}".format((training_loss.item())))

                if train_step not in train_steps_for_val or train_step == 0:

                    train_anchor_embeddings, train_positive_embeddings, train_negative_embeddings = triplet_net(training_data_batch)
                    training_loss = triplet_net.compute_triplet_loss(train_anchor_embeddings, train_positive_embeddings, train_negative_embeddings)

                    train_loss_accumulator += training_loss.item()

                    training_loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                else:
                    print()
                    triplet_net.eval()

                    val_loss_accumulator = 0
                    for val_step, validation_data_batch in enumerate(validation_dataloader):

                        with torch.no_grad():

                            val_anchor_embeddings, val_positive_embeddings, val_negative_embeddings = triplet_net(validation_data_batch)
                            validation_loss = triplet_net.compute_triplet_loss(val_anchor_embeddings, val_positive_embeddings, val_negative_embeddings)

                            val_loss_accumulator += validation_loss.item()

                            print('Trial Num : ' + str(trial_num) + ' Epoch: ' + str(epoch) + ' Step: ' + str(val_step) + ' Validation Loss: ' + "{:.8f}".format((validation_loss.item())))

                    print()
                    average_train_loss = np.round(train_loss_accumulator / train_steps_for_val, 8)
                    average_val_loss = np.round(val_loss_accumulator / len(validation_dataloader), 8)

                    train_loss_list.append(average_train_loss)
                    val_loss_list.append(average_val_loss)

                    train_loss_accumulator = 0  # Reset train loss accumulator
                    scheduler.step()

                    if average_val_loss < current_val_loss:
                        current_val_loss = average_val_loss

                        patience_counter = 0

                        try:
                            os.remove(outmodel_name)
                        except:
                            pass 

                        torch.save(triplet_net.state_dict(), outmodel_name)

                    else:
                        patience_counter += 1

                    print('Trial Num : ' + str(trial_num) + ' Patience Counter: ' + str(patience_counter) + ' Average Training Loss : ' + str(average_train_loss))
                    print('Trial Num : ' + str(trial_num) + ' Patience Counter: ' + str(patience_counter) + ' Average Validation Loss : ' + str(average_val_loss))
                    print()

                    if patience_counter >= max_patience_counter:
                        patience_exceeded = True  # Set the flag to indicate patience exceeded
                        break  # Break the inner loop
                        
        if patience_exceeded:
            break

    final_epoch = epoch
    final_train_step = train_step

    min_acheived_val_loss = min(val_loss_list)

    if min_acheived_val_loss <= study_best:

        triplet_net.eval()
        test_loss_accumulator = 0
        for test_step, testing_data_batch in enumerate(testing_dataloader):

            with torch.no_grad():

                test_anchor_embeddings, test_positive_embeddings, test_negative_embeddings = triplet_net(testing_data_batch)            
                testing_loss = triplet_net.compute_triplet_loss(test_anchor_embeddings, test_positive_embeddings, test_negative_embeddings)

                test_loss_accumulator += testing_loss.item()

                print('Trial Num : ' + str(trial_num) + ' Epoch: ' + str(epoch) + ' Step: ' + str(test_step) + ' testing Loss: ' + "{:.8f}".format((testing_loss.item())))

        average_test_loss = test_loss_accumulator / len(testing_dataloader) 
        print()
        print('Trial Num : ' + str(trial_num) + ' Epoch: ' + str(epoch) + '  Average Testing Loss: ' + "{:.8f}".format((average_test_loss)))
        print()

        dump(train_loss_list, 'TrainLossCurve.joblib')
        dump(val_loss_list, 'ValLossCurve.joblib')

        shutil_dir = trials_dir + 'Trial_' + str(parameter_dictionary['trial_num']) + '/'
        shutil_list = ['TrainLossCurve.joblib', 'ValLossCurve.joblib', outmodel_name]
        try:
            os.mkdir(shutil_dir)
            for i in shutil_list:
                shutil.move(i, shutil_dir)

        except:
            pass

    else:
        os.remove(outmodel_name)

    return min_acheived_val_loss, final_epoch, final_train_step

def objective(trial):

    seed_torch()
    trial_num = trial.number

    parameters = {
        
            'n_linear_layers' : trial.suggest_int('n_linear_layers', 1, 20, step=1),
            'decay_rate' : trial.suggest_float('decay_rate', 0.5, 0.9, log=False),
            'embedding_dim' : trial.suggest_int('embedding_dim', 8, 128, step=4),
            'activation_function' : trial.suggest_categorical('activation_function', ['RELU', 'GELU', 'SILU', 'ELU', 'Tanh', 'Sigmoid', 'Linear']),
            'dropout_weight' : trial.suggest_float('dropout_weight', 0.2, 0.75, log=False),

            'initial_learning_rate' : trial.suggest_float('initial_learning_rate', 0.0005, 0.01, log=False),
            'gamma' : trial.suggest_float('gamma', 0.1, 0.90, log=False),
            'batch_size' : trial.suggest_int('batch_size', 320, 512, step=32),
            'epochs' : 2,
                        
            'trial_num' : trial_num,
            }

    print('\n')
    print(parameters)
    print()

    trial_triplet_loss, trial_epoch, trial_train_step = train_model(parameters)

    trial.set_user_attr('final_epoch', trial_epoch)
    trial.set_user_attr('final_train_step', trial_train_step)

    return trial_triplet_loss

def scale_embedings(tr_df, tr_index, val_df, val_index,  te_df, te_index):

    scaler = StandardScaler()
    scaler.fit(tr_df)

    tr_scaled = pd.DataFrame(scaler.transform(tr_df))
    val_scaled = pd.DataFrame(scaler.transform(val_df))
    te_scaled = pd.DataFrame(scaler.transform(te_df))

    tr_scaled.index = tr_index
    val_scaled.index = val_index
    te_scaled.index = te_index

    dump(scaler, 'TripletNet_scaler.joblib')

    return tr_scaled, val_scaled, te_scaled

if __name__ == "__main__":

    cwd = os.getcwd() + '/' # Current working directory
    triplets_dir = '/'.join(cwd.split('/')[0:-2]) + '/' # Directory where triplet files are
    embedding_dir = '/'.join(cwd.split('/')[0:-4]) + '/' # Directory where embedding files are

    training_triplets = load(triplets_dir + 'Sentauri_DeepLoc_Training_TripletTraining.joblib') # Load training triplets
    validation_triplets = load(triplets_dir + 'Sentauri_DeepLoc_Training_TripletValidation.joblib') # Load validation triplets
    testing_triplets = load(triplets_dir + 'Sentauri_DeepLoc_Training_TripletTesting.joblib') # Load testing triplets

    training_accessions = list(set([item for sublist in training_triplets for item in sublist])) # Get training accession
    validation_accessions = list(set([item for sublist in validation_triplets for item in sublist])) # Get validation accessions
    testing_accessions = list(set([item for sublist in testing_triplets for item in sublist])) # Get testing accesions

    total_df = load(embedding_dir + 'Sentauri_DeepLoc_ProtBERT_Embeddings.joblib') # Load ProtBERT embeddings
    total_df = total_df[[i for i in range(1024)]] # Cull to only features

    training_df = total_df.loc[training_accessions] # Get training embeddings
    validation_df = total_df.loc[validation_accessions] # Get validation embeddings
    testing_df = total_df.loc[testing_accessions] # Get testing embeddings

    training_df_scaled, validation_df_scaled, testing_df_scaled = scale_embedings(training_df, training_accessions, validation_df, validation_accessions,  testing_df, testing_accessions)

    number_trials = 30
    number_jobs = 10

    study = optuna.create_study(direction='minimize',sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=number_trials, n_jobs=number_jobs, show_progress_bar=True)
    best_params = study.best_trial.params

    best_params['best_trial_num'] = study.best_trial.number
    best_params['epochs'] = study.best_trial.user_attrs['final_epoch']
    best_params['final_train_step'] = study.best_trial.user_attrs['final_train_step']

    print(best_params)

    dump(best_params, 'BestHyperparameters_TripletNetwork.joblib')
    study.trials_dataframe().to_csv('OptunaData.csv', index=False)

    dump(study, 'OptunaStudy.joblib')