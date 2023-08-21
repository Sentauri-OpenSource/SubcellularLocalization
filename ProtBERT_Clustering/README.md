# ProtBERT Clustering Directory

## File Description

- cluster_sequences_dbscan.py:
  - A Python script which will cluster the ProtBERT embeddings and then sample clusters to make ML datasplits.
  - Recall the embeddings file this code will act on (ProtBERT_Embeddings.joblib) is produced from get_protbert_embeddings.py   
  - This script assumes ProtBERT_Embeddings.joblib is in the previous working directory. Please make a new dir called MakeSplits with this script in it.  

## Running this file


```python  
    ipython 

    run cluster_sequences_dbscan.py  
```

