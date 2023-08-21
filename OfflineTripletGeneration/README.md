# Triplet Generation Directory

## File Description

- make_triplets_offline.py:
  - A Python script which will make triplets for all ML datasplits.
  - Recall the embeddings file this code will act on (ProtBERT_Embeddings_Training.joblib, ProtBERT_Embeddings_Validation.joblib, ad ProtBERT_Embeddings_Testing.joblib) are produced from cluster_sequences_dbscan.py  
  - This script assumes (ProtBERT_Embeddings_Training.joblib, ProtBERT_Embeddings_Validation.joblib, ad ProtBERT_Embeddings_Testing.joblib) are in the previous working directory. Please make a new dir called MakeOfflineTriplets with this script in it.  

## Running this file


```python  
    ipython 

    run make_triplets_offline.py  
```
