import os
import pandas as pd
from UCPModels import run_cases

def runAll(base_dir):
    #resources_dir = os.path.join(os.path.dirname(__file__), r'datasetA')
    p = os.path.abspath('..')
    file_path_train = os.path.join(p + '/datasetA' + "/" + base_dir, 'training.csv')
    file_path_test = os.path.join(p + '/datasetA' + "/" + base_dir, 'testing.csv')
    ds_train = pd.read_csv(file_path_train)
    ds_test = pd.read_csv(file_path_test)

    results = []
    print('--------------------------')
    kq = run_cases(ds_train, ds_test, base_dir)
    results.append(kq)
    return results
