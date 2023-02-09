# %%
from pymfe.mfe import MFE
import os
from scipy.io.arff import loadarff
import pandas as pd
import numpy as np
from tqdm import tqdm
import time
import warnings
from interruptingcow import timeout
warnings.filterwarnings('ignore')

# %%
prev_dir = '/users/guest/j/jhiggin6/Documents/Thesis/datasets'
my_dir = '/users/guest/j/jhiggin6/Documents/Thesis/pymfe/datasets_pymfe'
pymfe_dir = '/users/guest/j/jhiggin6/Documents/Thesis/pymfe'

# %%
if not os.path.exists(my_dir):
    for i in range(1, 467):
        if i == 147 or i == 157 or i == 387:
            continue
        if os.getcwd() != my_dir:
            if os.path.exists(my_dir):
                os.chdir(my_dir)
            else:
                os.mkdir(my_dir)
                os.chdir(my_dir)
        if not os.path.isdir(f'{my_dir}/dataset_{i}'):
            os.mkdir(f'dataset_{i}')
        os.chdir(f'dataset_{i}')
        raw_data = loadarff(f'{prev_dir}/data_{i}.arff')
        df = pd.DataFrame(raw_data[0])
        target = df.pop(df.iloc[:,-1].name)
        df = pd.get_dummies(df)
        mapping = {a:i for i, a in enumerate(list(target.unique()))}
        target = target.map(mapping)
        target
        df.to_csv(f"dataset_{i}_py.dat", index=False, header=False)
        target.to_csv(f"dataset_{i}_labels_py.dat", index=False, header=False)

# %%
def doStuff(X, y, i, groups):
        mfe = MFE(groups)
        mfe.fit(X, y)
        return mfe.extract()

def get_meta_features(groups, time_limit=30.0):
    datasets_finished = []
    datasets_failed = []

    for i in tqdm(range(1, 467), desc=groups):
        if i == 147 or i == 157 or i == 387:
            continue

        if i != 147 or i != 157 or i != 387:
            X = np.array(pd.read_csv(f"{my_dir}/dataset_{i}/dataset_{i}_py.dat", header=None))
            y = np.squeeze(np.array(pd.read_csv(f"{my_dir}/dataset_{i}/dataset_{i}_labels_py.dat", header=None)))

        start_time = time.time()
        try:  
            with timeout(time_limit, exception=RuntimeError):
                ft = doStuff(X, y, i, groups)
        except RuntimeError: 
            datasets_failed.append(i)
            end_time = time.time()
            continue

        end_time = time.time()
        mf = pd.DataFrame(columns=ft[0])
        try:
            mf.loc[0] = ft[1]
        except:
            datasets_failed.append(i)
            end_time = time.time()
        mf.to_csv(f"{my_dir}/dataset_{i}/dataset_{i}_{groups}_mfe.dat", index=False, header=True)
        datasets_finished.append(i)

    if not os.path.isdir(f'{pymfe_dir}/datasets_failed'):
        os.mkdir(f'{pymfe_dir}/datasets_failed')
    with open(f'{pymfe_dir}/datasets_failed/datasets_failed_{groups}', 'w') as file:
        file.write(f"Failed Datasets with {groups} MFE features:\n")
        for data_i in datasets_failed:
            file.write(f"{data_i}\n")

# %%
mfe = MFE()
possible_groups = list(mfe.valid_groups())
possible_groups.append("all")
possible_groups.append("default")
possible_groups
# print('possible_groups:\n', possible_groups)

# %%
# get_meta_features(groups='concept')       ##TODO: FIX THIS RUN
get_meta_features(groups='all')
get_meta_features(groups='default')

# %%
# import re
# failed = []
# myfile = open(f'{pymfe_dir}/datasets_failed/datasets_failed_model-based', "r")
# while myfile:
#     line  = myfile.readline()
#     failed.append(line)
#     if line == "":
#         break
# myfile.close() 
# failed.pop(0)
# failed.pop(-1)
# mylist = [int(re.search('(.+)\\n', t).group(1)) for t in failed]
# mylist
# for val in mylist:
#     os.remove(f'{pymfe_dir}/466datasets_pymfe/dataset_{val}/dataset{val}_model-based_mfe.dat')