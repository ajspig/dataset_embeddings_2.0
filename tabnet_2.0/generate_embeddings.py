from pytorch_tabnet.tab_network import EmbeddingGenerator
import torch
import arff
from scipy.io import arff
import pandas as pd
import numpy as np
import os
import re
from sklearn.preprocessing import LabelEncoder

# when we run this script I want to pass in what the cat_emb_dim is 
# possibly the folder for the dataset, but hold on that thought 

def get_tensor_file(filename):
    # create a pandas dataframe from the ARFF file
    with open(filename, 'r') as f:
        dataset = arff.loadarff(f)
    df = pd.DataFrame(dataset[0])
    input_dim = df.shape[1]

    # label encode all the categorical variables (not sure if this is the best practice)
    le = LabelEncoder()
    for col in df.select_dtypes(include=['object']):
        df[col] = le.fit_transform(df[col])
    
    tensor_data = torch.tensor(df.values)

    return input_dim, tensor_data

def read_process_arff(filename):
    data, meta = arff.loadarff(filename)
    attr_names = meta.names()
    attr_types = meta.types()
    cat_index = []
    cat_dims = []

    for i in range(len(attr_types)):
        if attr_types[i] == 'nominal':
            cat_index.append(i)
            num_cats = len(meta[attr_names[i]][1])
            cat_dims.append(num_cats)

    return cat_index, cat_dims

    

def main():    
    # Set random seed for reproducibility
    torch.manual_seed(0)
    np.random.seed(0)

    for filename in os.listdir('datasets'):
        if filename.endswith('.arff'):
            print(filename)
            # get the tensor file and the cat_index and cat_dims
            input_dim, tensor_data = get_tensor_file(os.path.join('datasets', filename))
            cat_index, cat_dims = read_process_arff(os.path.join('datasets', filename))
            # create the embedding generator
            embedding_generator = EmbeddingGenerator(input_dim=input_dim, cat_dims=cat_dims, cat_idxs=cat_index, cat_emb_dim=1)
            
            # embeddings = embedding_generator(tensor_data)
            embeddings = []
            try:
                embeddings = embedding_generator(tensor_data)
            except:
                print(input_dim, cat_dims, cat_index)
            if embeddings != []:
                # save the embeddings to a file 
                t_np = embeddings.detach().numpy()
                df = pd.DataFrame(t_np)
                #df.to_csv(os.path.join('embeddings', filename.replace('.arff', '.csv')), index=False)
                df.to_csv(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'embeddings', filename.replace('.arff', '.csv')), index=False)



if __name__ == "__main__":
    main()