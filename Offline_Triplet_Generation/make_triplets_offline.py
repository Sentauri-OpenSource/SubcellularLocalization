import os
import random
import numpy as np 
import pandas as pd
import torch 
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from joblib import dump, load
from tqdm import tqdm

''' Globals '''
seed=42
def seed_torch():
    
    ''' This function will set all possible seeds to produce deterministic behavior '''

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    return None

seed_torch() # Set seeds

dependent_variables = ['Cytoplasm', 'Nucleus', 'Extracellular', 'Cell_Membrane', 'Endoplasmic_Reticulum', 'Golgi_Apparatus', 'Mitochondria'] # DeepLoc dependant variables 

def make_triplets(x, y):

    '''
        This function takes in a vector x (uniprot accessions) and a dataframe y which correspond to the DeepLoc dependant variables.
        This function will assess how many subcellular localizations each protein sequence exhibits. 
        Based on this, it will determine how many negative samples are needed to make per anchor. 
        All samples in the set will be used as an anchor.
    '''

    def scan_positives():

        '''
        This function will find protein accessions with exact subcellular localizations: including multiples.
        Then, it will select positives at random and if the number of possibilities permits, it will not resample. 
        '''

        mask = np.logical_and.reduce([y[col] == val for col, val in zip(y.columns, anchor_subcellular_localizations)]) # Find rows with identical subcellular localization vectors
        positives_df = y[mask].dropna() # Filter data frame to these rows

        len_positive_samples = len(positives_df) # Find number of sequences with exact subcellular localizations. For determination of resampling
        selected_positives = [] # Store positives for triplets later

        positive_samples = positives_df.index.to_list() # Get all exact samples in a list
        for _ in range(n_possible_negative_examples): # Select positives based on number of needed negatives

            Positive = np.random.choice(positive_samples) # Random positive selection
            selected_positives.append(Positive) # Store positive selection

            if len_positive_samples <= n_possible_negative_examples:
                pass # Positive samples is small so we need to resample

            else:
                del positive_samples[positive_samples.index(Positive)] # Positive samples is large, so we will delete to not resample

        return selected_positives # Return chosen positives

    def scan_negatives():

        '''
        This function will find protein accessions with subcellular localizations completely different than the anchor.
        This includes removing any samples with any localizations in common with anchor.
        Then, it will select negatives at random according to the number of needed negatives from below. 
        '''

        negatives_df = y[y.iloc[:, [i for i, val in enumerate(anchor_subcellular_localizations) if val == 1]].sum(axis=1) == 0] # Find only rows which have no common 1 indexes in common
        negative_samples = negatives_df.index.to_list() # Store these accessions in a list

        selected_negatives = [] # Store negatives
        anchor_triplet_negative_classes = [] # List to break while loop condition and ensure that each subcellular location is sampled
        while len(anchor_triplet_negative_classes) != n_possible_negative_examples: # Break when the list reachs the correct length accoring to number negative samples

            Negative = np.random.choice(negative_samples) # Make random negative selection
            negative_df = pd.DataFrame(y.loc[Negative]).T # Get subcellular localizations for random negative
            other_subcellular_localizations = negative_df.columns[negative_df.eq(1).any()].to_list() # Find the alternate subcellular localizations and store in list

            for other_subcell_loc in other_subcellular_localizations: # Iterate all alternate subcellular localizations
                if other_subcell_loc in anchor_triplet_negative_classes: # If this one has already been seen, pass
                    pass 
                else:
                    anchor_triplet_negative_classes.append(other_subcell_loc) # Otherwise append this to the list
                    selected_negatives.append(Negative) # Store the negative
                    break # Break as to not to repeat accession if having multiple alternate locations

        return selected_negatives # Return chosen negatives

    triplets = [] # Store made triplets

    for depvar in dependent_variables: # Iterate all dependant variables
        print(depvar) # For sanity while waiting
        tmp_y = y[[depvar]][depvar].to_numpy() # Get individual depvar as numpy array

        for anchor_idx in tqdm(range(len(tmp_y))): # Iterate the depvar to get anchors

            Anchor = x[anchor_idx] # Get anchor by taking accession from x vector at anchor_idx
            anchor_df = pd.DataFrame(y.loc[Anchor]).T # Get anchor df to see all subcellular localizations
            anchor_subcellular_localizations = anchor_df.loc[Anchor].to_list() # Store localizations in list

            n_anchor_subcellular_localizations = int(sum(anchor_subcellular_localizations)) # Get number of localizations
            n_possible_negative_examples = len(dependent_variables) - n_anchor_subcellular_localizations # Compute how many neatives to get based on how many localizations are present

            positive_examples = scan_positives() # Get positive examples
            negative_examples = scan_negatives() # Get negative examples

            for example_index in range(n_possible_negative_examples): # Iterate the number of needed negatives
                triplets.append([Anchor, positive_examples[example_index], negative_examples[example_index]]) # Make and store the triples

    return triplets # Return the triplets 

if __name__ == "__main__":

    cwd = os.getcwd() + '/' # Current working directory
    data_dir = '/'.join(cwd.split('/')[0:-2]) + '/' # Directory one dir out where embedding files are 

    triplet_training_samples = load(data_dir + 'ProtBERT_Embeddings_Training.joblib')[dependent_variables]
    triplet_validation_samples = load(data_dir + 'ProtBERT_Embeddings_Validation.joblib')[dependent_variables]
    triplet_testing_samples = load(data_dir + 'ProtBERT_Embeddings_Testing.joblib')[dependent_variables]

    X_train = make_triplets(triplet_training_samples.index.to_list(), triplet_training_samples) # Make training set triplets
    X_valid = make_triplets(triplet_validation_samples.index.to_list(), triplet_validation_samples) # Make validation set triplets
    X_test = make_triplets(triplet_testing_samples.index.to_list(), triplet_testing_samples) # Make testing set triplets

    dump(X_train, 'TripletsTraining.joblib') # Use joblib to save training set triplets
    dump(X_valid, 'TripletsValidation.joblib') # Use joblib to save validation set triplets
    dump(X_test, 'TripletsTesting.joblib') # Use joblib to save testing set triplets