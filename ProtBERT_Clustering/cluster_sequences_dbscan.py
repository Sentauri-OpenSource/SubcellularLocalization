import os
import random
import numpy as np 
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import davies_bouldin_score
import torch
import umap
import optuna
import matplotlib.pyplot as plt
from joblib import load, dump 

seed=42
def seed_torch():
    
    ''' This function sets all library seeds to ensure determinism '''

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    return None

seed_torch() # Set all seeds

def cluster(params):

    ''' This function will cluster the ProtBERT embeddings and use a custom metric of goodness. It takes in a dictionary called params determined by the optuna trial '''

    trial_num = params['trial_num'] # Get optuna trial number

    if trial_num == 0:
        study_best = 1000000 # Set study best to high value if frist trial
        
    else:
        study_best = study.best_trial.values[0] # Otherwise get stored best value

    dbscan = DBSCAN(eps=params['eps'], min_samples=params['min_samples'], p=params['p']) # Instantiate the sklearn DBSCAN model with hyperparameters from params dict
    clusters = dbscan.fit_predict(protbert_df_scaled) # Fit and predict the data

    n_clusters = len(np.unique(clusters)) # Compute the number of found clusters
    if n_clusters == 1: # Ensure if only one cluster is found to return a large value
        return 1000000
    
    elif max([(np.count_nonzero(clusters == i) / len(protbert_df_scaled)) for i in range(-1, n_clusters -1)]) > 0.5: # Ensure if any one cluster contains more than 50% of the data, return a large value
        return 1000000

    else:
        db_score = davies_bouldin_score(protbert_df_scaled, clusters) # Otherwise compute the sklearn davies-bouldin score
        count = np.count_nonzero(clusters == -1) # Compute the size of the noise cluster given by the label -1

        score = count * db_score # Custom metric of the product of davies-bouldin and size of noise cluster. Both should be minimized

        if score < study_best: # If new best print useful data store cluster assignments
            print('DB Score = ' + str(db_score))
            print('Population Noise Cluster = ' + str(count))
            print('Number of Clusters = ' + str(n_clusters))
            dump(clusters, 'Optimal_DBSCAN_Clusters.joblib')

        return score # Return score to obective function

def objective(trial):

    seed_torch() # Seed everything
    trial_num = trial.number # Get trial number

    parameters = {
        
            'eps' : trial.suggest_float('eps', 1, 15, log=False),
            'min_samples' : trial.suggest_int('min_samples', 2, 100, step=1),
            'p' : trial.suggest_int('p', 2, 8, step=1),
            'trial_num' : trial_num
            } # Parameters dictionary

    print() # For visual appeal
    print(parameters) # Print parameters 
    print() # For visual appeal

    score = cluster(parameters) # Retrieve clustering score

    return score # Send score to optuna

def make_ml_sets(df):

    ''' This function will use the cluster assignments from tuning to make the training, validation, and testing sets for ML '''

    total_samples = len(df) # Find total number of sequences

    val_target_percent = 0.1 # Set validation percentage
    val_tolerance_percent = 0.11 # Set validation allowable tolerance percentage
    test_target_percent = 0.1 # Set testing percentage
    test_tolerance_percent = 0.11 # Set testing allowable tolerance percentage

    clusters = df['Cluster'].unique() # Find number of unique clusters

    val_target = int(total_samples * val_target_percent) # Convert percentage to number as int
    val_tolerance = int(total_samples * val_tolerance_percent) # Same for the tolerance percent

    test_target = int(total_samples * test_target_percent) # Convert percentage to number as int
    test_tolerance = int(total_samples * test_tolerance_percent) # Same for the tolerance percent

    train_set = pd.DataFrame(columns=df.columns) # Make empty training df
    val_set = pd.DataFrame(columns=df.columns) # Make empty validation df
    test_set = pd.DataFrame(columns=df.columns) # Make empty testing df

    train_set = pd.concat([train_set, df[df['Cluster'] == -1]]) # Add noise cluster to train set
    clusters = clusters[1:] # Remove noise cluster from available clusters

    np.random.shuffle(clusters) # Shuffle clusters

    for cluster in clusters: # Iterate clusters
        cluster_samples = df[df['Cluster'] == cluster] # Get sampled cluster
        cluster_size = len(cluster_samples) # Find size of sampled cluster

        current_len_val_set = len(val_set) # Get current size of the validation set
        proposed_size = current_len_val_set + cluster_size # Chek how large val set would be if sampled cluster is added
        
        if proposed_size <= val_target: # If proposed size is smaller than the target, add to the val df and delete that cluster
            val_set = pd.concat([val_set, cluster_samples])
            clusters = np.delete(clusters, np.where(clusters == cluster))

        elif proposed_size > val_target and proposed_size <= val_tolerance: # If proposed size is larger than the target, but smaller than tolerance, add to the val df, delete that cluster and break out
            val_set = pd.concat([val_set, cluster_samples])
            clusters = np.delete(clusters, np.where(clusters == cluster))
            break

        else: # Otherwise break
            break

    for cluster in clusters: # Do the same for the testing set
        cluster_samples = df[df['Cluster'] == cluster]
        cluster_size = len(cluster_samples)

        current_len_test_set = len(test_set)
        proposed_size = current_len_test_set + cluster_size
        
        if proposed_size <= test_target:
            test_set = pd.concat([test_set, cluster_samples])
            clusters = np.delete(clusters, np.where(clusters == cluster))

        elif proposed_size > test_target and proposed_size <= test_tolerance:
            test_set = pd.concat([test_set, cluster_samples])
            clusters = np.delete(clusters, np.where(clusters == cluster))
            break

        else:
            break

    for cluster in clusters: # Populate the training set with the remaining clusters
        cluster_samples = df[df['Cluster'] == cluster]
        train_set = pd.concat([train_set, cluster_samples])
        clusters = np.delete(clusters, np.where(clusters == cluster))

    train_set = train_set.drop(['Cluster'], axis=1) # Drop cluster col
    val_set = val_set.drop(['Cluster'], axis=1) # Drop cluster col
    test_set = test_set.drop(['Cluster'], axis=1) # Drop cluster col

    train_set.index.name = 'name' # Set index name to name
    val_set.index.name = 'name' # Set index name to name
    test_set.index.name = 'name' # Set index name to name

    train_set = pd.concat([train_set, labels.loc[train_set.index.to_list()]], axis=1) # Add dependent variable columns back in according to accession names
    val_set = pd.concat([val_set, labels.loc[val_set.index.to_list()]], axis=1) # Add dependent variable columns back in according to accession names
    test_set = pd.concat([test_set, labels.loc[test_set.index.to_list()]], axis=1) # Add dependent variable columns back in according to accession names

    dump(train_set, 'ProtBERT_Embeddings_Training.joblib') # Save file
    dump(val_set, 'ProtBERT_Embeddings_Validation.joblib') # Save file
    dump(test_set, 'ProtBERT_Embeddings_Testing.joblib') # Save file

    return None

if __name__ == "__main__":

    dependent_variables = ['Cytoplasm', 'Nucleus', 'Extracellular', 'Cell_Membrane', 'Endoplasmic_Reticulum', 'Golgi_Apparatus', 'Mitochondria'] # Labels of interest

    cwd = os.getcwd() + '/' # Current working directory
    protbert_emmbedding_dir = '/'.join(cwd.split('/')[0:-2]) + '/' # Previous directory

    protbert_data_file = protbert_emmbedding_dir + 'ProtBERT_Embeddings.joblib' # File containing ProtBERT embeddings
    protbert_df = load(protbert_data_file) # Load the data

    labels = protbert_df[dependent_variables] # Store the dependent variable columns
    protbert_df = protbert_df.drop(dependent_variables, axis=1) # Drop the dependent variable columns from the rest of the features

    scaler = StandardScaler() # Instantiate sklearn standard scaler
    protbert_df_scaled = pd.DataFrame(scaler.fit_transform(protbert_df)) # Fit an transform entire set
    protbert_df_scaled.index = protbert_df.index.to_list() # Store accessions to list

    number_trials = 1000 # Set number of Optuna trials

    study = optuna.create_study(direction='minimize',sampler=optuna.samplers.TPESampler(seed=seed)) # Create Optuna study wth direction minimize, TPESampler, and seed set
    study.optimize(objective, n_trials=number_trials, n_jobs=1, show_progress_bar=True) # Run the tuning 1 job at a time
    best_params = study.best_trial.params # Retreive best params

    dump(best_params, 'DBSCAN_Params.joblib') # Save best params
    best_clusters = load('Optimal_DBSCAN_Clusters.joblib') # Load the saved cluster assignments from tuning

    protbert_df_scaled['Cluster'] = best_clusters # Add Cluster column with cluster assignments to protbert_df_scaled 
    make_ml_sets(protbert_df_scaled) # Make training, validation, and testing sets from cluster sampling