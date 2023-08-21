import os 

if __name__ == "__main__":

    cwd = os.getcwd() + '/'
    models_dir = cwd + 'Models/'

    try:
        os.mkdir(models_dir)
    except:
        pass

    dependant_variables = ['Golgi_Apparatus', 'Endoplasmic_Reticulum', 'Cytoplasm', 'Nucleus', 'Extracellular', 'Cell_Membrane', 'Mitochondria']  # DeepLoc dependent variables
    script_name = 'train_catboost_models_concat.py'

    for idx, depvar in enumerate(dependant_variables):

        os.system('python ' + script_name + ' ' + str(idx) + ' ' + models_dir)
