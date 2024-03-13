from complexity import Complexity
import os
import csv
from multiprocessing import Pool
import time

# folds_folder = "/Users/miguel_cruz/Library/CloudStorage/OneDrive-UniversidadedeCoimbra/CV_Kfold/arff_datasets_folds"
folds_folder = "/Users/miguel_cruz/Documents/Miguel_Cruz/LEI/CISUC/CrossValidation/src/CV_algorithms/pyCV/new_partitions"

# partioned_files_path = "/Users/miguel_cruz/Documents/Miguel_Cruz/LEI/CISUC/CrossValidation/src/CV_algorithms/pyCV/processed_files.json"
# partitioned_files = {"/Users/miguel_cruz/Library/CloudStorage/OneDrive-UniversidadedeCoimbra/CV_Kfold/arff_datasets_folds": ["housevotes.arff", "wdbc.arff", "heart.arff", "saheart.arff", "crx.arff", "haberman.arff", "spectfheart.arff", "tic-tac-toe.arff", "monk-2.arff", "pima.arff", "breast.arff", "titanic.arff", "australian.arff", "mushroom.arff", "spambase.arff", "chess.arff", "banana.arff", "sonar.arff", "bupa.arff", "phoneme.arff", "wisconsin.arff", "bands.arff", "hepatitis.arff", "german.arff", "mammographic.arff", "ionosphere.arff", "appendicitis.arff"]}
# partitioned_files = {"/Users/miguel_cruz/Library/CloudStorage/OneDrive-UniversidadedeCoimbra/CV_Kfold/arff_datasets_folds": ["mushroom.arff","titanic.arff", "housevotes.arff", "breast.arff", "sonar.arff", "hepatitis.arff"]}

# partitioned_files = {"/Users/miguel_cruz/Library/CloudStorage/OneDrive-UniversidadedeCoimbra/CV_Kfold/arff_datasets_folds": ["titanic.arff", "housevotes.arff", "breast.arff", "sonar.arff", "ionosphere.arff"]}
partitioned_files = {"/Users/miguel_cruz/Documents/Miguel_Cruz/LEI/CISUC/CrossValidation/src/CV_algorithms/pyCV/new_partitions": ["caesarian-cat.arff", "hepatitis-cat.arff", "immunotherapy-cat.arff", "broadway2-cat.arff", "schizo-cat.arff", "student-g-cat.arff", "veteran-cat.arff", "traffic-cat.arff", "lymphography-v1-cat.arff", "cryotherapy-cat.arff", "servo-cat.arff", "fertility-diagnosis-cat.arff", "pharynx-1year-cat.arff", "creditscore-cat.arff", "lymphography-normal-fibrosis-cat.arff", "student-cg-cat.arff", "kidney-cat.arff", "broadwaymult0-cat.arff", "Edu-Data-HvsL-cat.arff", "pbc-cat.arff", "student-p-cat.arff", "icu-cat.arff", "cyyoung-cat.arff", "pharynx-3year-cat.arff", "heart-statlog-cat.arff", "cleveland-cat.arff", "pharynx-status-cat.arff", "broadwaymult3-cat.arff", "caesarian.arff", "cryotherapy.arff", "hepatitis.arff", "immunotherapy.arff", "creditscore.arff", "broadway3.arff", "broadway2.arff", "fertility-diagnosis.arff", "schizo.arff", "student-g.arff", "traffic.arff", "veteran.arff", "lymphography-v1.arff", "student-p.arff", "kidney.arff", "servo.arff", "lymphography-normal-fibrosis.arff", "pharynx-1year.arff", "cyyoung.arff", "pharynx-status.arff", "pharynx-3year.arff", "icu.arff", "heart-statlog.arff", "Edu-Data-HvsL.arff", "broadwaymult0.arff", "pbc.arff", "cleveland.arff", "broadwaymult6.arff", "broadwaymult4.arff", "broadwaymult3.arff", "broadwaymult5.arff", "glioma16.arff", "solvent.arff", "colon32.arff", "lupus.arff", "leukemia.arff", "appendicitis.arff", "bc-coimbra.arff", "breast-car.arff", "wine-1vs2.arff", "somerville.arff", "iris0.arff", "relax.arff", "parkinson.arff", "sonar.arff", "glass1.arff", "wpbc.arff", "ecoli_0_vs_1.arff", "newthyroid1.arff", "prnn_synth.arff", "hepato-PHvsALD.arff", "spectf.arff", "poker_9_vs_7.arff", "ecoli-0-1-3-7_vs_2-6.arff"]}
# partitioned_files = {"/Users/miguel_cruz/Documents/Miguel_Cruz/LEI/CISUC/CrossValidation/src/CV_algorithms/pyCV/new_partitions": ["poker_9_vs_7.arff"]}

results_folder = "results2"

def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def process_dataset(folds_folder, dataset_name, n_splits, reps, partition_strategy, metrics):
    dataset_name = dataset_name.split(".")[0]
    print(dataset_name, n_splits, partition_strategy)
    for partition_iteration in range(1, reps+1):
        for i in range(1, n_splits+1):
            train_split_path = os.path.join( folds_folder ,dataset_name+"-"+partition_strategy, dataset_name+"-"+partition_strategy+"-"+str(partition_iteration)+"-"+str(n_splits)+"-"+str(i)+"tra"+".arff")
            test_split_path = os.path.join( folds_folder ,dataset_name+"-"+partition_strategy, dataset_name+"-"+partition_strategy+"-"+str(partition_iteration)+"-"+str(n_splits)+"-"+str(i)+"tst"+".arff")
            
            complexity_train = Complexity(train_split_path,distance_func="default",file_type="arff")
            complexity_test = Complexity(test_split_path,distance_func="default",file_type="arff")

            for metric in metrics:
                # if(metric[0]=="N4"):
                #     print(train_split_path + " " + test_split_path)
                with open(os.path.join(results_folder, f'{n_splits}_folds_{partition_strategy}' ,f'{dataset_name}_{n_splits}_{partition_strategy}_{metric[0]}.csv'), 'a', newline='') as file:
                    writer = csv.writer(file)
                    if partition_iteration == 1 and i == 1:
                        writer.writerow(["partition_iteration", "split", f"train_{metric[0]}" , f"test_{metric[0]}"])
                    # print(metric[0], i)
                    result_train = getattr(complexity_train, metric[0])()
                    result_test = getattr(complexity_test, metric[0])()
                    if metric[1]:
                        result_train = max(result_train)
                        result_test = max(result_test)

                    writer.writerow([partition_iteration, i, result_train, result_test])

def delete_csv_files(results_folder, dataset_name, n_splits, partition_strategy, metrics):
    for metric in metrics:
        file_path =os.path.join(results_folder, f'{n_splits}_folds_{partition_strategy}' ,f'{dataset_name}_{n_splits}_{partition_strategy}_{metric[0]}.csv')
        # print(file_path)
        if os.path.exists(file_path):
            os.remove(file_path)

def process(args):
    folds_folder, dataset_name, n_splits, reps, partition_strategy, metrics = args
    delete_csv_files(os.path.join(results_folder, f'{n_splits}_folds_{partition_strategy}'), dataset_name.split(".")[0], n_splits, partition_strategy, metrics)
    process_dataset(folds_folder, dataset_name, n_splits, reps, partition_strategy=partition_strategy, metrics=metrics)

def main():
    n_splits = 5
    reps = 50
    partition_strategies = ["SCV", "DBSCV", "DOBSCV", "MSSCV",]
    feature_metrics = [('F1', 1), ('F1v', 1), ('F2', 1), ('F3', 1), ('F4', 1), ('input_noise', 1)] 
    structural_metrics = [('N1', 0),  ('T1', 0), ('Clust', 0), ('DBC', 0), ('LSC', 0), ('NSG', 0)] #('N2', 0),
    instance_metrics = [('R_value', 1), ('deg_overlap', 0), ('CM', 0), ('kDN', 0), ('N4', 0), ('N3', 0), ('SI', 0), ('borderline', 0)]  
    multiresolution_metrics = [ ('C1', 0), ('C2', 0)] 
    all_metrics = feature_metrics + structural_metrics + instance_metrics + multiresolution_metrics

    args_list = []
    for partition_strategy in partition_strategies:
            create_directory(os.path.join(results_folder, f'{n_splits}_folds_{partition_strategy}'))
    start_time = time.time()
    for folds_folder in partitioned_files.keys():
        for dataset_name in partitioned_files[folds_folder]:
            for partition_strategy in partition_strategies[:]:
                # args = (folds_folder, dataset_name, n_splits, reps, partition_strategy, all_metrics)
                # process(args)
                args_list.append((folds_folder, dataset_name, n_splits, reps, partition_strategy, all_metrics))

    with Pool(8) as p:
        p.map(process, args_list)
    end_time = time.time()
    print("Time elapsed normal: ", end_time - start_time)

if __name__ == "__main__":
    main()



# ------------------- Not optimal -------------------

# def process_dataset(folds_folder, dataset_name, n_splits, reps, partition_strategy, metric):
#     dataset_name = dataset_name.split(".")[0]
#     # print(f"Processing {dataset_name} with {n_splits} splits and {reps} repetitions using {partition_strategy} partition strategy")
#     with open(os.path.join(results_folder, f'{n_splits}_folds_{partition_strategy}' ,f'{dataset_name}_{n_splits}_{partition_strategy}_{metric[0]}.csv'), 'w', newline='') as file:
#         writer = csv.writer(file)
#         writer.writerow(["partition_iteration", "split", f"train_{metric[0]}" , f"test_{metric[0]}"])
#         for partition_iteration in range(1, reps+1):
#             for i in range(1, n_splits+1):
#                 train_split_path = os.path.join( folds_folder ,dataset_name+"-"+partition_strategy, dataset_name+"-"+partition_strategy+"-"+str(partition_iteration)+"-"+str(n_splits)+"-"+str(i)+"tra"+".arff")
#                 test_split_path = os.path.join( folds_folder ,dataset_name+"-"+partition_strategy, dataset_name+"-"+partition_strategy+"-"+str(partition_iteration)+"-"+str(n_splits)+"-"+str(i)+"tst"+".arff")

#                 complexity_train = Complexity(train_split_path,distance_func="default",file_type="arff")
#                 complexity_test = Complexity(test_split_path,distance_func="default",file_type="arff")

#                 result_train = getattr(complexity_train, metric[0])()
#                 result_test = getattr(complexity_test, metric[0])()
#                 if metric[1]:
#                     result_train = max(result_train)
#                     result_test = max(result_test)

#                 # if isinstance(result_train, list):
#                 #     print(f"Train: {metric[0]} is a list")
#                 writer.writerow([partition_iteration, i, result_train, result_test])
        

# #----------------- No Threads -----------------
# # def main():
# #     n_splits = 5
# #     reps = 10
# #     partition_strategies = ["SCV", "MSSCV", "DBSCV", "DOBSCV"]
# #     feature_metrics = [('F1', 1), ('F1v', 1), ('F2', 1), ('F3', 1), ('F4', 1), ('input_noise', 1)] 
# #     structural_metrics = [('N1', 0), ('N2', 0), ('T1', 0), ('Clust', 0), ('DBC', 0), ('LSC', 0), ('NSG', 0)] 
# #     instance_metrics = [('R_value', 1), ('deg_overlap', 0), ('CM', 0), ('kDN', 0), ('N4', 0), ('N3', 0), ('SI', 0), ('borderline', 0)]  # IPoints, wCM, dwCM 
# #     multiresolution_metrics = [ ('C1', 0), ('C2', 0), ('purity', 0), ('neighbourhood_separability', 0)] 

# #     all_metrics = feature_metrics + structural_metrics + instance_metrics + multiresolution_metrics


# #     for folds_folder in partitioned_files.keys():
# #         for dataset_name in partitioned_files[folds_folder]:
# #             for partition_strategy in partition_strategies[:1]:
# #                 for metric in all_metrics:
# #                     process_dataset(folds_folder, dataset_name, n_splits, reps, partition_strategy=partition_strategy, metric=metric)


# #     feature_metrics_dump = []
# #     structural_metrics_dump = [('ONB_avg', 0), ('ONB_tot', 0), ('ICSV', 0)]
# #     instance_metrics_dump = [('D3_value', 0),]
# #     multiresolution_metrics_dump = [('MRCA', 0),('purity', 0), ('neighbourhood_separability', 0)]



# #----------------- Threads -----------------
# def process(args):
#         folds_folder, dataset_name, n_splits, reps, partition_strategy, metric = args
#         process_dataset(folds_folder, dataset_name, n_splits, reps, partition_strategy=partition_strategy, metric=metric)


# def main():
#     n_splits = 5
#     reps = 200
#     partition_strategies = ["SCV", "MSSCV", "DBSCV", "DOBSCV"]
#     feature_metrics = [('F1', 1), ('F1v', 1), ('F2', 1), ('F3', 1), ('F4', 1), ('input_noise', 1)] 
#     structural_metrics = [('N1', 0), ('N2', 0), ('T1', 0), ('Clust', 0), ('DBC', 0), ('LSC', 0), ('NSG', 0)] 
#     instance_metrics = [('R_value', 1), ('deg_overlap', 0), ('CM', 0), ('kDN', 0), ('N4', 0), ('N3', 0), ('SI', 0), ('borderline', 0)]  # IPoints, wCM, dwCM 
#     multiresolution_metrics = [ ('C1', 0), ('C2', 0), ('purity', 0), ('neighbourhood_separability', 0)] 
#     all_metrics = feature_metrics + structural_metrics + instance_metrics + multiresolution_metrics

#     args_list = []
#     for folds_folder in partitioned_files.keys():
#         for dataset_name in partitioned_files[folds_folder]:
#             for partition_strategy in partition_strategies[:]:
#                 for metric in all_metrics:
#                     args_list.append((folds_folder, dataset_name, n_splits, reps, partition_strategy, metric))

#     with Pool(10) as p:
#         p.map(process, args_list)


# if __name__ == "__main__":
#     main()