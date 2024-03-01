import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
import csv
import os

folder_results_path = "/Users/miguel_cruz/Documents/Miguel_Cruz/LEI/CISUC/CrossValidation/src/CV_complexity/pycol/results"


# partitioned_files = {"/Users/miguel_cruz/Library/CloudStorage/OneDrive-UniversidadedeCoimbra/CV_Kfold/arff_datasets_folds": ["housevotes.arff", "wdbc.arff", "heart.arff", "saheart.arff", "crx.arff", "haberman.arff", "spectfheart.arff", "tic-tac-toe.arff", "monk-2.arff", "pima.arff", "breast.arff", "titanic.arff", "australian.arff", "mushroom.arff", "spambase.arff", "chess.arff", "banana.arff", "sonar.arff", "bupa.arff", "phoneme.arff", "wisconsin.arff", "bands.arff", "hepatitis.arff", "german.arff", "mammographic.arff", "ionosphere.arff", "appendicitis.arff"]}
partitioned_files = {"/Users/miguel_cruz/Library/CloudStorage/OneDrive-UniversidadedeCoimbra/CV_Kfold/arff_datasets_folds": ["housevotes.arff", "breast.arff", "sonar.arff", "bupa.arff", "wisconsin.arff", "hepatitis.arff", "appendicitis.arff"]}
# partitioned_files = {"/Users/miguel_cruz/Library/CloudStorage/OneDrive-UniversidadedeCoimbra/CV_Kfold/arff_datasets_folds": ["housevotes.arff"]}



def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def join_dataset_metrics(folder, dataset_name, partition_strategy, n_splits, output_folder):
    results_path = os.path.join(folder, f"{n_splits}_folds_{partition_strategy}")
    csv_files = [f for f in os.listdir(results_path) if f.endswith('.csv') and dataset_name in f and  f.split('_')[2] == partition_strategy and "all" not in f]
    df_list = [ pd.read_csv(os.path.join(results_path, csv_file), usecols=lambda column : column not in ["partition_iteration", "split"]) for csv_file in csv_files[1:]]
    concatenated_df = pd.concat([pd.read_csv(os.path.join(results_path, csv_files[0]))] + df_list, axis=1)
    concatenated_df.to_csv(os.path.join(output_folder, f"{dataset_name}_{n_splits}_{partition_strategy}_all.csv"), index=False)

def get_standard_deviation(folder, dataset_name, partition_strategy, n_splits, output_folder):
    # results_path = os.path.join(folder, f"{n_splits}_folds_{partition_strategy}")
    all_file_path = os.path.join(folder, f"{dataset_name}_{n_splits}_{partition_strategy}_all.csv")
    all_file = pd.read_csv(all_file_path)

    std_per_fold = all_file.iloc[:, 2:].std()    
    std_per_split = pd.DataFrame()
    for i in range(2, len(std_per_fold), 2):
        std_per_split[f"{std_per_fold.index[i].split('_')[1]}_std"] = all_file.iloc[:, i:i+2].std(axis=1)
    std_per_split.loc['mean'] = std_per_split.mean()
    std_per_split.to_csv(os.path.join(output_folder, f"{dataset_name}_{n_splits}_{partition_strategy}_std.csv"), index=True)



def main():
    # dataset_name = "housevotes"
    n_splits = 5
    partition_strategies = ["SCV", "MSSCV", "DBSCV", "DOBSCV"]


    for folds_folder in partitioned_files.keys():
        for dataset_name in partitioned_files[folds_folder]:
            dataset_name = dataset_name.split(".")[0]
            for partition_strategy in partition_strategies[:1]:
                results_analysis_path = os.path.join(folder_results_path, f"{dataset_name}_metrics", f"{n_splits}_folds_{partition_strategy}")
                create_directory(results_analysis_path)

                join_dataset_metrics(folder_results_path, dataset_name=dataset_name, partition_strategy=partition_strategy, n_splits=n_splits, output_folder=results_analysis_path)
                get_standard_deviation(results_analysis_path, dataset_name=dataset_name, partition_strategy=partition_strategy, n_splits=n_splits, output_folder=results_analysis_path)

if __name__ == "__main__":
    main()

            
