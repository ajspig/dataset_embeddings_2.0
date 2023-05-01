"""parses arff files into variables that are passed into pickle files that are then read into data_helper_covertype.py"""

import re
import pickle

# loop through the directory of datasets (all 466) and replace the file with the arr_file path 
# maybe I have a sperate file that runs this 

# specify the path to the ARFF file
arff_file = "path/to/your/arff/file.arff"

# initialize variables
NUM_SAMPLES = 0
NUM_FEATURES = 0
LABEL_COLUMN = ""
BOOL_COLUMNS = []
INT_COLUMNS = []
STR_COLUMNS = []
FLOAT_COLUMNS = []

# read in the ARFF file
with open(arff_file, "r") as f:
    lines = f.readlines()

# iterate over the lines in the ARFF file
for line in lines:
    # check for the number of samples
    if line.startswith("@data"):
        NUM_SAMPLES = len(lines) - lines.index(line) - 1
    # check for attribute declarations
    elif line.startswith("@attribute"):
        # extract the attribute name and type
        attr_name, attr_type = re.findall(r"@attribute\s+([\w\d_-]+)\s+(\w+)", line)[0]
        # check the type of the attribute
        if attr_type.lower() == "numeric":
            INT_COLUMNS.append(attr_name)
        elif attr_type.lower() == "string":
            STR_COLUMNS.append(attr_name)
        elif attr_type.lower() == "real":
            FLOAT_COLUMNS.append(attr_name)
        elif attr_type.lower() == "{true,false}":
            BOOL_COLUMNS.append(attr_name)
        # check for the label column
        elif attr_type.lower() == "nominal":
            LABEL_COLUMN = attr_name
            values = re.findall(r"{(.+)}", line)[0]
            values = [val.strip() for val in values.split(",")]
            num_classes = len(values)
            str_nuniquess = values
            # count the number of features
            NUM_FEATURES += 1

# print out the extracted information
print(f"Number of samples: {NUM_SAMPLES}")
print(f"Number of features: {NUM_FEATURES}")
print(f"Label column: {LABEL_COLUMN}")
print(f"Boolean columns: {BOOL_COLUMNS}")
print(f"Integer columns: {INT_COLUMNS}")
print(f"String columns: {STR_COLUMNS}")
print(f"Floating-point columns: {FLOAT_COLUMNS}")

# change 'variables.pkl' to be a name specific to the dataset 
with open('variables.pkl', 'wb') as f:
    pickle.dump((NUM_SAMPLES, NUM_FEATURES, LABEL_COLUMN, BOOL_COLUMNS, INT_COLUMNS, STR_COLUMNS, FLOAT_COLUMNS), f)