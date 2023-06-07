# this script will run get_model_acc.py with its appropriate arguments for each embedding

import os
import sys
import subprocess
import csv
import numpy as np
import pandas as pd
from random import sample

# script arguments:
# k-fold number
# path to the meta-dataset csv folder
# path to the embeddings folder

# read in the meta-dataset csv folder
