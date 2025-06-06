import os
import re
import csv
import pandas as pd
import argparse
from pprint import pprint
from datasets import load_dataset
import numpy as np
from sklearn.model_selection import train_test_split


def check_data(filebase):

    file_train = filebase+'_train.csv'
    file_valid = filebase+'_valid.csv'
    #file_test = filebase+'_test.csv'
    
    dataset_files = {
        "train": [file_train],
        "validation": [file_valid],
        #"test": [file_test],
    }
    dataset1 = load_dataset("csv", data_files=dataset_files)
    print(dataset1)

    
def main(filename):
    
    filename_base, ext = os.path.splitext(filename)
    print(f'file:{filename} -> filebase:{filename_base}')
    
    df = pd.read_csv(filename, delimiter="\t")

    d_train, d_test_valid = train_test_split(df, test_size=0.2, random_state=1) #80% 20%に分割
    d_test, d_valid = train_test_split(d_test_valid, test_size=0.5, random_state=1) # 20%を半分に分割(10%/10%)
    file_train = filename_base+'_train.csv'
    file_valid = filename_base+'_valid.csv'
    file_test = filename_base+'_test.csv'    
    print(f'training data : {d_train.shape} ==> {file_train}')
    print(f'validation data : {d_valid.shape} ==> {file_valid}')    
    print(f'test data : {d_test.shape} ==> {file_test}')
    d_train.to_csv(file_train)
    d_valid.to_csv(file_valid)
    d_test.to_csv(file_test)    
    
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('-f', '--file', required=False)    # Target Data file        
    parser.add_argument('-split', '--split', action='store_true')    # flag for data preparation(split to train/valid/test)
    parser.add_argument('-check', '--check', action='store_true')    # flag for data check
    parser.add_argument('-filebase', '--filebase', required=False)    # Target Data file    
    args = parser.parse_args()

    tgt_file = args.file
    filebase = args.filebase

    if args.split:
        main(tgt_file)
    elif args.check:
        print(f'data check...')
        check_data(filebase)
