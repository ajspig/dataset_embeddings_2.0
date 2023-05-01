import pandas as pd
from scipy.io.arff import loadarff
import os

prev_dir = '/Users/joshhiggins/Documents/Grad_School/thesis/466datasets'
my_dir = '/Users/joshhiggins/Documents/Grad_School/thesis/dataset2vec-2/466datasets'


def main():

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

            mapping = {a:i for i, a in enumerate(list(target.unique()))} # I think this might be the problem 
            target = target.map(mapping)

            df.to_csv(f"dataset_{i}_py.dat", index=False, header=False)
            target.to_csv(f"dataset_{i}_labels_py.dat", index=False, header=False)


    

if __name__ == "__main__":
    main()