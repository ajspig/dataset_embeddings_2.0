import pandas as pd
import numpy as np
import os
from sklearn.model_selection import KFold
import csv
import sys
import csv
from random import sample
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold
from tqdm import tqdm

# script arguments: 
# k-fold number 
# path to the meta-dataset csv file

def read_dataset(meta_dataset_path):
    results = []
    with open(meta_dataset_path, 'r') as file:
        reader = csv.reader(file)
        for count, row in enumerate(reader):
            # if its the first row then we save it as col_names and dont add to results
            if count == 0:
                col_names = row[1:]
            else:
                results.append(row[1:])
    
    return pd.DataFrame(results, columns=col_names)

def create_dataset(meta_dataset_path, maxes):
    df = pd.read_csv(meta_dataset_path)
    # not sure why below is needed but it is, also not sure if this is the best appproach
    # TODO: come back to 
    df.replace(np.nan, 0, inplace=True)
    df.replace(np.inf, 0, inplace=True)

    X = []
    y = []
    valid_datasets = []

    # built like this becuase not all the datsets work with each embedding so just using the index doesnt work 

    for i in range(df.shape[0]):
        # check the first item in the ith row of df
        if df.loc[i, 'dataset'] in maxes:
            y.append(sample(maxes[df.loc[i, 'dataset']],k=1)[0])
            X.append(df.iloc[i, 1:].tolist())
            valid_datasets.append(df.loc[i, 'dataset'])
    X = np.array(X)
    y = np.array(y)
    valid_datasets = np.array(valid_datasets)

    # shuffle the datsets bfore calling train_meta_model
    X_y = np.append(X, y.reshape(-1,1), axis=1)
    everything = np.append(X_y, valid_datasets.reshape(-1,1), axis=1)
    np.random.shuffle(everything)
    valid_datasets = everything[:, -1]
    y = everything[:, -2]
    X = everything[:, :-2]
    return X, y, valid_datasets


def train_meta_model(sk_algorithm, X, y, valid_datasets, maxes, kfold_num=10):
	accuracies = []
	kf = KFold(n_splits=kfold_num)
	kf.get_n_splits(X)
	for train_index, test_index in tqdm(kf.split(X)):
		X_train, X_test = X[train_index], X[test_index]
		y_train, y_test = y[train_index], y[test_index] # dont use y_test
		ds_train, ds_test = valid_datasets[train_index], valid_datasets[test_index] # valid_datasets is the list of datasets
		sk_algorithm.fit(X_train, y_train)
		y_pred = sk_algorithm.predict(X_test)
		correct = 0
		for ds, prediction in zip(ds_test, y_pred):
			if prediction in maxes[ds]:
				correct += 1
		accuracies.append(correct/len(ds_test))
	return max(accuracies) # do I want to return accuracies max?

def main(meta_dataset_path, kfold_num):
    kfold_num = int(kfold_num)    
    dirpath = os.path.dirname(os.path.realpath(__file__))
    os.chdir(dirpath)
    maxes = {}

    # make this a command line argument (if we use optim or default )
    with open('optim_maxes.csv', 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            key = int(row[0])
            values_str = ','.join(row[1:])
            values_str = values_str.strip()
            values = list(map(int, values_str.split(',')))
            maxes[key] = values
    #meta_dataset_path = "/users/guest/a/as2273/research_spring/dataset_embeddings_2.0/analyze_meta_models/meta_datasets/concept_metadataset.csv"
    X, y, valid_datasets = create_dataset(meta_dataset_path, maxes)

    results_arr = []
        # Random Forest
    rf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
    rf_accuracies = train_meta_model(rf, X, y, valid_datasets, maxes, kfold_num=kfold_num)
    results_arr.append(np.mean(rf_accuracies))
    print(f'rf accuracy: {round(np.mean(rf_accuracies)*100,2)}%', flush=True)

    # Logistic Regression
    lr = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial', max_iter=100000)
    lr_accuracies = train_meta_model(lr, X, y, valid_datasets, maxes, kfold_num=kfold_num)
    results_arr.append(np.mean(lr_accuracies))
    print(f'lr accuracy: {round(np.mean(lr_accuracies)*100,2)}%', flush=True)

    # SVM
    svm = SVC(gamma='auto')
    svm_accuracies = train_meta_model(svm, X, y, valid_datasets, maxes, kfold_num=kfold_num)
    results_arr.append(np.mean(svm_accuracies))
    print(f'svm accuracy: {round(np.mean(svm_accuracies)*100,2)}%', flush=True)

    # KNN
    knn = KNeighborsClassifier(n_neighbors=3)
    knn_accuracies = train_meta_model(knn, X, y, valid_datasets, maxes, kfold_num=kfold_num)
    results_arr.append(np.mean(knn_accuracies))
    print(f'knn accuracy: {round(np.mean(knn_accuracies)*100,2)}%', flush=True)

    # Naive Bayes
    nb = GaussianNB()
    nb_accuracies = train_meta_model(nb, X, y, valid_datasets, maxes, kfold_num=kfold_num)
    results_arr.append(np.mean(nb_accuracies))
    print(f'nb accuracy: {round(np.mean(nb_accuracies)*100,2)}%')

    # Decision Tree
    dt = DecisionTreeClassifier(random_state=0)
    dt_accuracies = train_meta_model(dt, X, y, valid_datasets, maxes, kfold_num=kfold_num)
    results_arr.append(np.mean(dt_accuracies))
    print(f'dt accuracy: {round(np.mean(dt_accuracies)*100,2)}%')

    # AdaBoost
    ab = AdaBoostClassifier(n_estimators=100, random_state=0)
    ab_accuracies = train_meta_model(ab, X, y, valid_datasets, maxes, kfold_num=kfold_num)
    results_arr.append(np.mean(ab_accuracies))
    print(f'ab accuracy: {round(np.mean(ab_accuracies)*100,2)}%')

    # Gradient Boosting
    gb = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0,)
    gb_accuracies = train_meta_model(gb, X, y, valid_datasets, maxes, kfold_num=kfold_num)
    results_arr.append(np.mean(gb_accuracies))
    print(f'gb accuracy: {round(np.mean(gb_accuracies)*100,2)}%')

    # Neural Network
    nn = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1, max_iter=100000000)
    # print out and check all the types of what we pass into the train_meta_model function
    nn_accuracies = train_meta_model(nn, X, y, valid_datasets, maxes, kfold_num=kfold_num)
    results_arr.append(np.mean(nn_accuracies))
    print(f'nn accuracy: {round(np.mean(nn_accuracies)*100,2)}%')

    # Bagging
    bg = BaggingClassifier(estimator=DecisionTreeClassifier(), n_estimators=10, random_state=0)
    bg_accuracies = train_meta_model(bg, X, y, valid_datasets, maxes, kfold_num=kfold_num)
    results_arr.append(np.mean(bg_accuracies))
    print(f'bg accuracy: {round(np.mean(bg_accuracies)*100,2)}%')

    # Extra Trees
    et = ExtraTreesClassifier(n_estimators=100, max_depth=None, min_samples_split=2, random_state=0)
    et_accuracies = train_meta_model(et, X, y, valid_datasets, maxes, kfold_num=kfold_num)
    results_arr.append(np.mean(et_accuracies))
    print(f'et accuracy: {round(np.mean(et_accuracies)*100,2)}%')

    # Voting Classifier
    vc = VotingClassifier(estimators=[('rf', rf), ('lr', lr), ('svm', svm), ('knn', knn), ('nb', nb), ('dt', dt), ('ab', ab), ('gb', gb), ('nn', nn), ('bg', bg), ('et', et)], voting='hard')
    vc_accuracies = train_meta_model(vc, X, y, valid_datasets, maxes, kfold_num=kfold_num)
    results_arr.append(np.mean(vc_accuracies))
    print(f'vc accuracy: {round(np.mean(vc_accuracies)*100,2)}%')

    # save results_arr to a csv file named based off the meta_dataset_path
    # get the file name. it will be the last part of the path starts with / and ends with .csv
    file_name = meta_dataset_path.split('/')[-1]
    file_name = file_name.split('.')[0]
    # now save results_arr to a csv file
    with open(f'{file_name}_acc.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(results_arr)




if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])