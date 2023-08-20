# Subcellular Localization Data Directory  

Swiss-Prot_HumanSubset_SubcellularLocalization_08202023.csv is the main processed data file for which machine learning sets were made.  
SubcellularLocalizations_WithAccessions_08202023.json is a json file with all Swiss-Prot accessions organized according to subcellular localization.  
SubcellularLocalization_Counts_08202023.csv is a csv file containing the number of accessions in every subcellular localization.  

To load the csv files with pandas:  
  import pandas as pd  

  df = pd.read_csv('Swiss-Prot_HumanSubset_SubcellularLocalization_08202023.csv')  
  print(df.head(10))  

To load the json file:
  import json  
    
  with open('SubcellularLocalizations_WithAccessions_08202023.json', 'r') as fp:  
      data = json.load(fp)  
