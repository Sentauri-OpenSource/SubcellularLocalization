import os
import numpy as np 
import pandas as pd
from sklearn.exceptions import UndefinedMetricWarning
import torch
from torch_geometric.loader import DataLoader
from joblib import dump, load
from tqdm import tqdm
import warnings 
from utils import seed_torch, worker_init_fn
from models import TripletEmbeddingDataset, TripletNet

'''Globals'''
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="torch_geometric.utils.scatter")

cwd = os.getcwd() + '/'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
seed=42
seed_torch()

dependant_variables = ['Cytoplasm', 'Nucleus', 'Extracellular', 'Cell_Membrane', 'Endoplasmic_Reticulum', 'Golgi_Apparatus', 'Mitochondria'] # DeepLoc dependant variables 

def process(protbert_df_depvar, outname):

    total_accessions = protbert_df_depvar.index.to_list()
    total_triplets = [[i, i, i] for i in total_accessions]

    protbert_df = protbert_df_depvar[[i for i in range(1024)]]

    scaler = load(cwd + 'TripletNet_scaler.joblib')
    protbert_df_scaled = pd.DataFrame(scaler.transform(protbert_df))
    protbert_df_scaled.index = protbert_df_depvar.index.to_list()

    df_labels = protbert_df_depvar[dependant_variables]

    triplet_net = TripletNet(best_params).to(device)
    saved_model_path = best_trial_dir + 'DeepLoc_TripletNetwork_' + str(best_trial) + '.pt'
    triplet_net.load_state_dict(torch.load(saved_model_path))

    data_to_encode = TripletEmbeddingDataset(total_triplets, protbert_df_scaled, device)
    encoding_dataloader = DataLoader(data_to_encode, batch_size=best_params['batch_size'], shuffle=False, worker_init_fn=worker_init_fn)

    triplet_embeddings_holder = []

    triplet_net.eval()
    for step, data_batch in enumerate(tqdm(encoding_dataloader)):
        with torch.no_grad():

            triplet_embeddings, _, __ = triplet_net(data_batch)
            triplet_embeddings = triplet_embeddings.detach().cpu().numpy()

            triplet_embeddings_holder.append(triplet_embeddings)

    triplet_embeddings_holder = np.concatenate(triplet_embeddings_holder, axis=0)

    out_df = pd.DataFrame(triplet_embeddings_holder)
    out_df.index = protbert_df_scaled.index.to_list()
    out_df = pd.concat([out_df, df_labels], axis=1)
    out_df.index.name = 'ACC'

    dump(out_df, outname)

    return None

if __name__ == "__main__":

    best_params = load(cwd + 'BestHyperparameters_TripletNetwork.joblib')
    best_trial = best_params['best_trial_num']

    best_trial_dir = cwd + 'Trials/Trial_' + str(best_trial) + '/'

    data_dir = '/'.join(cwd.split('/')[0:-3]) + '/'

    training_data_file = data_dir + 'Sentauri_DeepLoc_ProtBERT_Embeddings_Training.joblib'
    validation_data_file = data_dir + 'Sentauri_DeepLoc_ProtBERT_Embeddings_Validation.joblib'

    training_df = load(training_data_file)
    validation_df = load(validation_data_file)
    training_df = pd.concat([training_df, validation_df], axis=0)
    training_triplet_outname = 'Sentauri_DeepLocTraining_Triplet_Embeddings.joblib'
    process(training_df, training_triplet_outname)


    testing_data_file = data_dir + 'Sentauri_DeepLoc_ProtBERT_Embeddings_Testing.joblib'
    testing_df = load(testing_data_file)
    training_triplet_outname = 'Sentauri_DeepLocTesting_Triplet_Embeddings.joblib'
    process(testing_df, training_triplet_outname)