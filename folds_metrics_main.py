from complexity import Complexity
import os
import csv
from multiprocessing import Pool


folds_folder = "/Users/miguel_cruz/Library/CloudStorage/OneDrive-UniversidadedeCoimbra/CV_Kfold/arff_datasets_folds"
# partioned_files_path = "/Users/miguel_cruz/Documents/Miguel_Cruz/LEI/CISUC/CrossValidation/src/CV_algorithms/pyCV/processed_files.json"
# partitioned_files = {"/Users/miguel_cruz/Library/CloudStorage/OneDrive-UniversidadedeCoimbra/CV_Kfold/arff_datasets_folds": ["housevotes.arff", "wdbc.arff", "heart.arff", "saheart.arff", "crx.arff", "haberman.arff", "spectfheart.arff", "tic-tac-toe.arff", "monk-2.arff", "pima.arff", "breast.arff", "titanic.arff", "australian.arff", "mushroom.arff", "spambase.arff", "chess.arff", "banana.arff", "sonar.arff", "bupa.arff", "phoneme.arff", "wisconsin.arff", "bands.arff", "hepatitis.arff", "german.arff", "mammographic.arff", "ionosphere.arff", "appendicitis.arff"]}
partitioned_files = {"/Users/miguel_cruz/Library/CloudStorage/OneDrive-UniversidadedeCoimbra/CV_Kfold/arff_datasets_folds": ["housevotes.arff", "breast.arff", "sonar.arff", "bupa.arff", "wisconsin.arff", "hepatitis.arff", "appendicitis.arff"]}
# partitioned_files = {"/Users/miguel_cruz/Library/CloudStorage/OneDrive-UniversidadedeCoimbra/CV_Kfold/arff_datasets_folds": ["housevotes.arff"]}


def process_dataset(folds_folder, dataset_name, n_splits, reps, partition_strategy, metric):
    dataset_name = dataset_name.split(".")[0]
    # print(f"Processing {dataset_name} with {n_splits} splits and {reps} repetitions using {partition_strategy} partition strategy")
    with open(os.path.join('results', f'{n_splits}_folds_{partition_strategy}' ,f'{dataset_name}_{n_splits}_{partition_strategy}_{metric[0]}.csv'), 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["partition_iteration", "split", f"train_{metric[0]}" , f"test_{metric[0]}"])
        for partition_iteration in range(1, reps+1):
            for i in range(1, n_splits+1):
                train_split_path = os.path.join( folds_folder ,dataset_name+"-"+partition_strategy, dataset_name+"-"+partition_strategy+"-"+str(partition_iteration)+"-"+str(n_splits)+"-"+str(i)+"tra"+".arff")
                test_split_path = os.path.join( folds_folder ,dataset_name+"-"+partition_strategy, dataset_name+"-"+partition_strategy+"-"+str(partition_iteration)+"-"+str(n_splits)+"-"+str(i)+"tst"+".arff")

                complexity_train = Complexity(train_split_path,distance_func="default",file_type="arff")
                complexity_test = Complexity(test_split_path,distance_func="default",file_type="arff")

                result_train = getattr(complexity_train, metric[0])()
                result_test = getattr(complexity_test, metric[0])()
                if metric[1]:
                    result_train = max(result_train)
                    result_test = max(result_test)

                # if isinstance(result_train, list):
                #     print(f"Train: {metric[0]} is a list")
                writer.writerow([partition_iteration, i, result_train, result_test])






            # writer.writerow(["partition_iteration", "split", metric[1]])
            # for partition_iteration in range(1, reps+1):
            #     for i in range(1, n_splits+1):
            #         # train_data, test_data = load_arff_fold(dataset_name=dataset_name, folds_folder=folds_folder, n_folds=n_splits, split_index=i, partition_iteration=partition_iteration, partition_strategy=partition_strategy)
            #         # classifier = classifier_info[0]
            #         # auc = run_fold(train_data, test_data, classifier)
            #         # total_auc += auc
            #         writer.writerow([partition_iteration, i, metric[0]])

    
    # open file to write results with this name, but having the metric name at the end of this file name



        

# def main():
#     n_splits = 5
#     reps = 10
#     partition_strategies = ["SCV", "MSSCV", "DBSCV", "DOBSCV"]
#     feature_metrics = [('F1', 1), ('F1v', 1), ('F2', 1), ('F3', 1), ('F4', 1), ('input_noise', 1)] 
#     structural_metrics = [('N1', 0), ('N2', 0), ('T1', 0), ('Clust', 0), ('DBC', 0), ('LSC', 0), ('NSG', 0), ('ICSV', 0)] 
#     instance_metrics = [('R_value', 1), ('deg_overlap', 0), ('CM', 0), ('D3_value', 0), ('kDN', 0), ('N4', 0), ('N3', 0), ('SI', 0), ('borderline', 0)]  # IPoints, wCM, dwCM 
#     multiresolution_metrics = [ ('C1', 0), ('C2', 0), ('purity', 0), ('neighbourhood_separability', 0)] 

#     all_metrics = feature_metrics + structural_metrics + instance_metrics + multiresolution_metrics


#     for folds_folder in partitioned_files.keys():
#         for dataset_name in partitioned_files[folds_folder]:
#             for partition_strategy in partition_strategies[:1]:
#                 for metric in all_metrics:
#                     process_dataset(folds_folder, dataset_name, n_splits, reps, partition_strategy=partition_strategy, metric=metric)


#     feature_metrics_dump = []
#     structural_metrics_dump = [('ONB_avg', 0), ('ONB_tot', 0)]
#     instance_metrics_dump = []
#     multiresolution_metrics_dump = [('MRCA', 0),]

def process(args):
        folds_folder, dataset_name, n_splits, reps, partition_strategy, metric = args
        process_dataset(folds_folder, dataset_name, n_splits, reps, partition_strategy=partition_strategy, metric=metric)


def main():
    n_splits = 5
    reps = 5
    partition_strategies = ["SCV", "MSSCV", "DBSCV", "DOBSCV"]
    feature_metrics = [('F1', 1), ('F1v', 1), ('F2', 1), ('F3', 1), ('F4', 1), ('input_noise', 1)] 
    structural_metrics = [('N1', 0), ('N2', 0), ('T1', 0), ('Clust', 0), ('DBC', 0), ('LSC', 0), ('NSG', 0), ('ICSV', 0)] 
    instance_metrics = [('R_value', 1), ('deg_overlap', 0), ('CM', 0), ('D3_value', 0), ('kDN', 0), ('N4', 0), ('N3', 0), ('SI', 0), ('borderline', 0)]  # IPoints, wCM, dwCM 
    multiresolution_metrics = [ ('C1', 0), ('C2', 0), ('purity', 0), ('neighbourhood_separability', 0)] 
    all_metrics = feature_metrics + structural_metrics + instance_metrics + multiresolution_metrics
    
    # def process(args):
    #     folds_folder, dataset_name, partition_strategy, metric = args
    #     process_dataset(folds_folder, dataset_name, n_splits, reps, partition_strategy=partition_strategy, metric=metric)

    args_list = []
    for folds_folder in partitioned_files.keys():
        for dataset_name in partitioned_files[folds_folder]:
            for partition_strategy in partition_strategies[:1]:
                for metric in all_metrics:
                    args_list.append((folds_folder, dataset_name, n_splits, reps, partition_strategy, metric))

    with Pool(10) as p:
        p.map(process, args_list)


if __name__ == "__main__":
    main()