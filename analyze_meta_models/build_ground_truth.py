# Goal: build a csv file with the ground truth for the datasets (from emp_results.csv)
# will only need to be run once
import pandas as pd
import numpy as np
import re
import os

def multiple_max(df):
    df1 = df.drop('dataset', axis=1)

    maxes = {}      # {dataset: [max1, max2, ...]}
    max_val = {}    # {dataset: max_val}
    for i in range(df.shape[0]):
        row = np.array(df1.iloc[i])
        row_maxes = np.argwhere(row == np.amax(row)).flatten().tolist()
        max_val[df.iloc[i, 0]] = np.amax(row)
        maxes[df.iloc[i, 0]] = row_maxes
        # maxes[df.iloc[i, 0]] = df1.columns[row_maxes].tolist() # if you want col names
    return maxes, max_val

dirpath = os.path.dirname(os.path.realpath(__file__))
os.chdir(dirpath)
pre = pd.read_csv("emp_results.csv")
pre['dataset'] = pre['dataset'].apply(lambda x: int(re.split(r'\.arff', x)[0]))

optimized_cols = []
default_cols = []

for col in pre.columns:
    if col == 'dataset':
        optimized_cols.append(col)
        default_cols.append(col)
    if "+" in col:
        optimized_cols.append(col)
    elif "-" in col:
        default_cols.append(col)

optim_maxes, optim_max_values = multiple_max(pre[optimized_cols])
default_maxes, default_max_values = multiple_max(pre[default_cols])

# optim_maxes = {dataset: [max1, max2, ..., max466]}
# default_maxes = {dataset: [max1, max2, ..., max466]}

# save optim_maxes and default_maxes to csv ( dont care about optim_max_values and default_max_values)
# optim_maxes_df = pd.DataFrame.from_dict(optim_maxes, orient='index')
# optim_maxes_df.to_csv("optim_maxes.csv")

# default_maxes_df = pd.DataFrame.from_dict(default_maxes, orient='index')
# default_maxes_df.to_csv("default_maxes.csv")

# save the dictionary to a csv file
# with open ('optim_maxes.csv', 'w') as f:
#     for key in optim_maxes.keys():
#         values = optim_maxes[key]
#         values_str = ','.join(str(i) for i in values)
#         f.write("%s,%s\n" % (key, values_str))
import csv
with open('optim_maxes.csv', 'w') as f:
    writer = csv.writer(f)
    for key, values in optim_maxes.items():
        writer.writerow([key] + values)
