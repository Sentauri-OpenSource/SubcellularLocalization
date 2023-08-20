# Subcellular Localization Data Directory

## Files Description

- Swiss-Prot_HumanSubset_SubcellularLocalization_08202023.csv:
  - The main processed data file for which machine learning sets were made.

- SubcellularLocalizations_WithAccessions_08202023.joblib:
  - A joblib file with all Swiss-Prot accessions organized according to subcellular localization.

- SubcellularLocalization_Counts_08202023.csv:
  - A CSV file containing the number of accessions in every subcellular localization.

## Loading CSV Files with Pandas


```python  
    import pandas as pd  

    df = pd.read_csv('Swiss-Prot_HumanSubset_SubcellularLocalization_08202023.csv')  
    print(df.head(10))    
```

## Loading Joblib File

```python
    from joblib import load  
    
    df = load('SubcellularLocalizations_WithAccessions_08202023.joblib')  
    print(df)  
```
