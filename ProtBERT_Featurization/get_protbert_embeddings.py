import os 
import random
import numpy as np
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from torch.utils.data import DataLoader
import datasets
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
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # Get GPU
batch_size = 32 # Will safely fit on A6000 GPU
dependant_variables = ['Cytoplasm', 'Nucleus', 'Extracellular', 'Cell_Membrane', 'Endoplasmic_Reticulum', 'Golgi_Apparatus', 'Mitochondria'] # DeepLoc dependant variables 

protbert_tokenizer = BertTokenizer.from_pretrained('Rostlab/prot_bert', do_lower_case=False) # Expects sequence to be uppercase
protbert_model = BertForSequenceClassification.from_pretrained('Rostlab/prot_bert', output_hidden_states=True).to(device) # Set up model to extract embeddings output_hidden_states=True

class Embeddings:

    '''
        This class will extract ProtBERT embeddings from a pandas dataframe of unprocessed uppercase protein sequences. 
        The get() method will preprocess and extract embeddings in a batched fashion.
        The output will be a pandas data frame of shape (dataset_length, 1024)
    '''

    def __init__(self, tmp_df, tokenizer, model, device, batch_size):

        self.tmp_df = tmp_df # Input dataframe
        self.tokenizer = tokenizer # ProtBERT tokenizer
        self.model = model # ProtBERT model
        self.device = device # GPU
        self.batch_size = batch_size # 64

    def get(self):

        ''' This function will take in a pandas dataframe with a single column called Sequence and will produce a data frame with ProtBERT embeddings '''

        def preprocess_data():

            ''' This function will take protein sequences, add spaces, tokenize, pad, and truncate '''

            sequences = self.tmp_df['Sequence'].to_list() # Store sequences in list
            spaced_sequences = [' '.join(list(i)) for i in sequences] # Add spaces between amino acids
            self.tmp_df['Sequence'] = spaced_sequences # overwrite sequences with spaced sequences

            dataset = datasets.Dataset.from_pandas(self.tmp_df) # Create dataset object from dataframe
            tokenized_dataset = dataset.map(lambda batch: self.tokenizer(batch['Sequence'], padding='max_length', truncation=True, max_length=2048), batched=True) # Tokenize dataset with padding to max length 1024 and truncate if longer
            tokenized_dataset.set_format("torch", columns=['input_ids', 'attention_mask']) # Set data to pytorch tensor

            return DataLoader(tokenized_dataset, batch_size=self.batch_size, shuffle=False) # Return DataLoader object with shuffle = False

        def extract_embeddings(batch):

            ''' This function will take in a batch from DataLoader and return ProtBERT embeddings '''

            input_ids = batch['input_ids'].to(self.device) # Get tokenized sequence from batch tuple and send to GPU
            attention_mask = batch['attention_mask'].to(self.device) # Get attention mask from batch tuple and send to GPU

            output = self.model(input_ids, attention_mask=attention_mask, output_hidden_states=True) # Pass data to ProtBERT model with output_hidden_states=True

            hidden_states = torch.stack(output.hidden_states).transpose(0, 1) # Make batch_size by 31 by max_seq_length by 1024
            hidden_states = torch.sum(hidden_states, dim=1) # Sum over all attention layers
            hidden_states = torch.mean(hidden_states, dim=1) # Average all tokens

            return hidden_states.detach().cpu().numpy() # Return embeddings to CPU
        
        tmp_dataloader = preprocess_data() # Get DataLoader to extact embeddings from input dataframe

        self.model.eval() # Eval so we dont update gradients
        total_embeddings = [] # Store whole dataloader embeddings

        with torch.no_grad(): # So we dont update gradients
            for batch in tqdm(tmp_dataloader): # Iterate batchs
                total_embeddings.append(extract_embeddings(batch)) # Extract and store batch embeddings

        return pd.DataFrame(np.concatenate(total_embeddings, axis=0)) # Return embeddings in dataframe format

def run():

    total_train_df = load('SubcellularLocalizations_WithAccessions_08202023.joblib') # Training set file
    train_df = total_train_df[['Sequence']] # Cull to accession and sequence columns
    total_train_df = total_train_df[dependant_variables] # Cull total dataframe to dependant variables for concatenation later

    train_embeddings = Embeddings(train_df, protbert_tokenizer, protbert_model, device, batch_size).get() # Instantiate and call get method of Embeddings class to get ProtBERT embeddings
    train_embeddings.index = train_df.index.to_list() # Set index with accessions from earlier. This works because DataLoader above is set to shuffle=False
    train_embeddings = pd.concat([train_embeddings, total_train_df], axis=1) # Concatenate embeddings with dependant variables
    train_embeddings.index.name = 'name' # Set pandas df index name to name
    dump(train_embeddings, 'ProtBERT_Embeddings.joblib') # Joblib save of pandas dataframe

    return None

if __name__ == '__main__':

    run() # Get training set ProtBERT embeddings