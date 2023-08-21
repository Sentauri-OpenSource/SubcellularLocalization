import os
import numpy as np
import pandas as pd
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import davies_bouldin_score
import torch
from torch_geometric.loader import DataLoader
from joblib import dump, load
from tqdm import tqdm
import warnings
from utils import seed_torch
import umap
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
import matplotlib.patches as mpatches

'''Globals'''
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="torch_geometric.utils.scatter")

cwd = os.getcwd() + '/'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
seed = 42
seed_torch()

dependant_variables = ['Cytoplasm', 'Nucleus', 'Extracellular', 'Cell_Membrane', 'Endoplasmic_Reticulum', 'Golgi_Apparatus', 'Mitochondria']  # DeepLoc dependent variables

if __name__ == "__main__":

    training_triplet_embeddings_df = load('Sentauri_DeepLocTraining_Triplet_Embeddings.joblib')
    training_labels_df = training_triplet_embeddings_df[dependant_variables]
    training_triplet_embeddings_df = training_triplet_embeddings_df.drop(dependant_variables, axis=1)

    testing_triplet_embeddings_df = load('Sentauri_DeepLocTesting_Triplet_Embeddings.joblib')
    testing_labels_df = testing_triplet_embeddings_df[dependant_variables]
    testing_triplet_embeddings_df = testing_triplet_embeddings_df.drop(dependant_variables, axis=1)

    training_len = len(training_triplet_embeddings_df)
    testing_len = len(testing_triplet_embeddings_df)

    total_triplet_df = pd.concat([training_triplet_embeddings_df, testing_triplet_embeddings_df], axis=0)
    total_labels_df = pd.concat([training_labels_df, testing_labels_df], axis=0)

    umap_obj = umap.UMAP()
    umap_embedding = umap_obj.fit_transform(total_triplet_df)

    for idx, depvar in enumerate(dependant_variables):

        outname = depvar.replace(' ', '')

        tmp_labels = total_labels_df[depvar].to_list()

        colors_training = [(0.0, 0.0, 1.0, 1.0), (1.0, 0.0, 0.0, 1.0)]
        colors_testing = [(0.0, 1.0, 0.0, 1.0), (1.0, 1.0, 0.0, 1.0)]

        label_colors_training = [colors_training[int(label)] for label in tmp_labels[:training_len]]
        label_colors_testing = [colors_testing[int(label)] for label in tmp_labels[training_len:training_len + testing_len]]

        plt.figure(figsize=(14, 6))  # Increase the figure size
        plt.subplots_adjust(wspace=0.3)  # Adjust the width spacing between subplots

        plt.subplot(1, 2, 1)  # First subplot for training data
        training_scatter = plt.scatter(umap_embedding[:training_len, 0], umap_embedding[:training_len, 1], c=label_colors_training)
        plt.title(depvar.replace('_', ' ').title() + ' Training Triplet Embeddings', fontsize=14)
        plt.xlabel('UMAP 1', fontsize=12)
        plt.ylabel('UMAP 2', fontsize=12)

        training_db_score = davies_bouldin_score(training_triplet_embeddings_df.values, training_labels_df[depvar].to_numpy())

        anchored_text = AnchoredText(f'Davies Bouldin: {training_db_score:.3f}', loc='lower left', frameon=False)
        plt.gca().add_artist(anchored_text)

        legend_handles_training = [mpatches.Patch(color=colors_training[0], label='Not Localized')]
        legend_handles_training.append(mpatches.Patch(color=colors_training[1], label='Localized'))
        plt.legend(handles=legend_handles_training, loc='upper right', fontsize=10)  # Legend inside the plot

        plt.subplot(1, 2, 2)  # Second subplot for testing data
        testing_scatter = plt.scatter(umap_embedding[training_len:, 0], umap_embedding[training_len:, 1], c=label_colors_testing)
        plt.title(depvar.replace('_', ' ').title() + ' Testing Triplet Embeddings', fontsize=14)
        plt.xlabel('UMAP 1', fontsize=12)
        plt.ylabel('UMAP 2', fontsize=12)

        testing_db_score = davies_bouldin_score(testing_triplet_embeddings_df.values, testing_labels_df[depvar].to_numpy())

        anchored_text = AnchoredText(f'Davies Bouldin: {testing_db_score:.3f}', loc='lower left', frameon=False)
        plt.gca().add_artist(anchored_text)

        legend_handles_testing = [mpatches.Patch(color=colors_testing[0], label='Not Localized')]
        legend_handles_testing.append(mpatches.Patch(color=colors_testing[1], label='Localized'))
        plt.legend(handles=legend_handles_testing, loc='upper right', fontsize=10)  # Legend inside the plot

        plt.tight_layout()
        plt.savefig('Training_And_Testing_UMAP_' + outname + '.png')
        plt.close()