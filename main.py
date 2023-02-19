import argparse
import glob
import json
import os
import pickle
from collections import Counter

import scipy.io
from joblib import Parallel, delayed
from sklearn.metrics import *
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from classes_list import *
from customized_statistical_tests import *
from tmfg_core import *

parser = argparse.ArgumentParser(description='TMFG Feature Selection.')
parser.add_argument('--stage',
                    type=str,
                    default='TFS',
                    choices=['SM_COMPUTATION', 'IFS', 'TFS', 'IFS_TEST', 'TFS_TEST', 'STATISTICAL_TEST'],
                    help="Stage to be run.")
parser.add_argument('--dataset',
                    type=str,
                    default='lung_small',
                    choices=['PCMAC', 'RELATHE', 'COIL20', 'ORL', 'warpAR10P', 'warpPIE10P',
                             'Yale', 'USPS', 'colon', 'GLIOMA', 'lung', 'lung_small', 'lymphoma',
                             'GISETTE', 'Isolet', 'MADELON'],
                    help="Dataset to be used for the experiments.")
parser.add_argument('--classification_algo',
                    type=str,
                    default='KNN',
                    choices=['KNN', 'LinearSVC', 'DecisionTree'],
                    help="Algorithm to be used during classification.")
parser.add_argument('--cc_type',
                    type=str,
                    default='pearson',
                    choices=['pearson', 'spearman'],
                    help="Type of correlation coefficient to be computed.")
parser.add_argument('--stat_test_pair',
                    type=str,
                    default='tfs_ifs',
                    help="Pair of classifiers to be statistically compared.")
parser.add_argument('--stat_test_setting',
                    type=str,
                    default='local',
                    help="Statistical comparison mode. DO NOT CHANGE.")
parser.add_argument('--test_mode',
                    type=str,
                    default='local',
                    help="Test mode. DO NOT CHANGE.")
args = parser.parse_args()


def get_mat_file_name(path):
    filename = os.path.splitext(os.path.basename(path))[0]
    return filename


def read_mat_files(path):
    mat = scipy.io.loadmat(path)
    X = mat['X'].astype(float)
    y = mat['Y'][:, 0]

    return X, y


def train_test_split_files(X, y, filename, data_dictionary):
    X, y = shuffle(X, y, random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)

    local_data_dictionary = {'X_train': X_train,
                             'X_test': X_test,
                             'y_train': y_train,
                             'y_test': y_test}

    data_dictionary[filename] = local_data_dictionary

    return data_dictionary


def get_data_description(data_dictionary, filename, description_dictionary):
    local_description_dictionary = {'#_features': data_dictionary[filename]['X_train'].shape[1],
                                    '#_samples_training': data_dictionary[filename]['X_train'].shape[0],
                                    '#_samples_test': data_dictionary[filename]['X_test'].shape[0],
                                    'counting_labels_training': Counter(data_dictionary[filename]['y_train']),
                                    'counting_labels_test': Counter(data_dictionary[filename]['y_test'])}

    description_dictionary[filename] = local_description_dictionary

    return description_dictionary


def get_data_files_extension(path):
    extension = os.path.splitext(os.path.basename(path))[1]
    return extension


def read_dexter_dataset(path, dataset_type, read_data_dictionary):
    data_dictionary = read_data_dictionary
    extension = get_data_files_extension(path)

    if extension == '.labels':
        y = np.loadtxt(path)

    vectors = np.zeros((300, 20000))

    if extension == '.data':

        with open(path, mode='r') as fid:
            data = fid.readlines()

        row = 0
        for line in data:
            line = line.strip().split()
            for word in line:
                col, val = word.split(':')
                vectors[row][int(col) - 1] = int(val)
            row += 1

        X = vectors

    if dataset_type == 'train' and extension == '.data':
        data_dictionary['X_train'] = X

    if dataset_type == 'train' and extension == '.labels':
        data_dictionary['y_train'] = y

    if dataset_type == 'valid' and extension == '.data':
        data_dictionary['X_test'] = X

    if dataset_type == 'valid' and extension == '.labels':
        data_dictionary['y_test'] = y

    return data_dictionary


def read_non_dexter_dataset(path, dataset_type, read_data_dictionary):
    data_dictionary = read_data_dictionary
    extension = get_data_files_extension(path)

    if extension == '.labels':
        y = np.loadtxt(path)
    else:
        X = np.loadtxt(path)

    if dataset_type == 'train' and extension == '.data':
        data_dictionary['X_train'] = X

    if dataset_type == 'train' and extension == '.labels':
        data_dictionary['y_train'] = y

    if dataset_type == 'valid' and extension == '.data':
        data_dictionary['X_test'] = X

    if dataset_type == 'valid' and extension == '.labels':
        data_dictionary['y_test'] = y

    return data_dictionary


def read_data_files(filename, paths_list, data_dictionary):
    read_data_dictionary = {}
    for path in paths_list:

        if ('train' in path) or ('valid' in path):

            dataset_type = 'train' if 'train' in path else 'valid'

            if filename == 'DEXTER':
                read_data_dictionary = read_dexter_dataset(path, dataset_type, read_data_dictionary)
            else:
                read_data_dictionary = read_non_dexter_dataset(path, dataset_type, read_data_dictionary)

    data_dictionary[filename] = read_data_dictionary

    return data_dictionary


def produce_correlation_matrix(data_dictionary, dataset_name, method):
    data = pd.DataFrame(data_dictionary['X_train']).fillna(method="ffill").fillna(method="bfill")
    data = data.loc[:, data.std() > 0.0]

    if method == 'spearman':
        data.corr(method='spearman').to_csv(f'spearman_{dataset_name}.csv', index=False)
    elif method == 'pearson':
        data.corr().to_csv(f'pearson_{dataset_name}.csv', index=False)


def hyper_opt_ifs(classification_algo, X_train, y_train, alpha, factor, num, dataset_name):
    np.random.seed(0)
    if classification_algo == 'LinearSVC':
        clf = LinearSVC(random_state=0, max_iter=50000)
    elif classification_algo == 'KNN':
        clf = KNeighborsClassifier()
    else:
        clf = DecisionTreeClassifier(random_state=0)

    pipeline = Pipeline([('ifs', IFS_class(num=num, dataset_name=dataset_name, alpha=alpha, factor=factor, step='cv')), ('scaling', StandardScaler()), ('clf', clf)])
    ifs_metric = cross_val_score(pipeline, X_train, y_train, cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=0), scoring='balanced_accuracy').mean()

    ifs_dictionary = {'alpha': round(alpha, 2),
                      'factor': round(factor, 2),
                      'num_features': num,
                      'score': round(ifs_metric, 2)}

    return ifs_dictionary


def ifs_pipeline(data_dictionary, dataset_name, classification_algo):
    data = pd.DataFrame(data_dictionary['X_train']).fillna(method="ffill").fillna(method="bfill")
    data = data.loc[:, data.std() > 0.0]
    data = data.to_numpy()

    alpha_values = np.arange(0.1, 1, 0.1)
    factor_values = np.arange(0.1, 1, 0.1)
    num_features = [10, 50, 100, 150, 200]

    output = Parallel(n_jobs=8)(
        delayed(hyper_opt_ifs)(classification_algo, data, data_dictionary['y_train'], alpha, factor, num, dataset_name) for alpha in alpha_values for factor in factor_values for
        num in num_features)
    output = sorted(output, key=lambda x: x['num_features'])

    output_file = open(f'./full_ifs_cv/{classification_algo}/{dataset_name}_ifs_full_cv.json', 'w', encoding='utf-8')
    for dic in output:
        json.dump(dic, output_file)
        output_file.write("\n")

    optimization = (max(output, key=lambda x: x['score']))
    print(optimization)
    cv_file = open(f"./optimal_ifs_cv/{classification_algo}/{dataset_name}_optimal_ifs_cv.pkl", "wb")
    pickle.dump(optimization, cv_file)
    cv_file.close()


def hyper_opt_tmfg(classification_algo, X_train, y_train, correlation_value, correlation_type, num, alpha, dataset_name):
    if correlation_value == 'energy' and correlation_type == 'square':
        return None
    else:
        if correlation_value == 'energy' and alpha != None:
            np.random.seed(0)
            if classification_algo == 'LinearSVC':
                clf = LinearSVC(random_state=0, max_iter=50000)
            elif classification_algo == 'KNN':
                clf = KNeighborsClassifier()
            else:
                clf = DecisionTreeClassifier(random_state=0)
            pipeline = Pipeline([('tfs', TFS_class(num=num, dataset_name=dataset_name, alpha=alpha, method=correlation_value, correlation_type=correlation_type, step='cv')),
                                 ('scaling', StandardScaler()), ('estimator', clf)])

            tmfg_metric = cross_val_score(pipeline, X_train, y_train, cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=0), scoring='balanced_accuracy').mean()

            tmfg_dictionary = {'correlation_value': correlation_value,
                               'correlation_type': correlation_type,
                               'num_features': num,
                               'alpha': alpha,
                               'score': round(tmfg_metric, 2)}

            return tmfg_dictionary

        elif correlation_value != 'energy' and alpha == None:
            np.random.seed(0)
            if classification_algo == 'LinearSVC':
                clf = LinearSVC(random_state=0, max_iter=50000)
            elif classification_algo == 'KNN':
                clf = KNeighborsClassifier()
            else:
                clf = DecisionTreeClassifier(random_state=0)
            pipeline = Pipeline([('tfs', TFS_class(num=num, dataset_name=dataset_name, alpha=alpha, method=correlation_value, correlation_type=correlation_type, step='cv')),
                                 ('scaling', StandardScaler()), ('estimator', clf)])
            tmfg_metric = cross_val_score(pipeline, X_train, y_train, cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=0), scoring='balanced_accuracy').mean()

            tmfg_dictionary = {'correlation_value': correlation_value,
                               'correlation_type': correlation_type,
                               'num_features': num,
                               'alpha': alpha,
                               'score': round(tmfg_metric, 2)}

            return tmfg_dictionary

        else:
            tmfg_dictionary = {'correlation_value': correlation_value,
                               'correlation_type': correlation_type,
                               'num_features': 10000000,
                               'alpha': alpha,
                               'score': 0}

            return tmfg_dictionary


def tmfg_pipeline(data_dictionary, dataset_name, classification_algo):
    data = pd.DataFrame(data_dictionary['X_train']).fillna(method="ffill").fillna(method="bfill")
    data = data.loc[:, data.std() > 0.0]
    data = data.to_numpy()

    correlation_values = ['pearson', 'spearman', 'energy']
    correlation_types = ['normal', 'square']
    num_features = [10, 50, 100, 150, 200]
    alpha_values = list(np.arange(0.1, 1, 0.1))
    alpha_values.append(None)

    output = Parallel(n_jobs=8)(
        delayed(hyper_opt_tmfg)(classification_algo, data, data_dictionary['y_train'], correlation_value, correlation_type, num_feature, alpha, dataset_name) for correlation_value
        in correlation_values for correlation_type in correlation_types for num_feature in num_features for alpha in alpha_values)
    output = [x for x in output if x is not None]
    output = sorted(output, key=lambda x: x['num_features'])

    list_cv = []
    for e in output:
        if e['num_features'] != 10000000:
            list_cv.append(e)
    del output

    output_file = open(f'./full_tfs_cv/{classification_algo}/{dataset_name}_tmfg_full_cv.json', 'w', encoding='utf-8')
    for dic in list_cv:
        json.dump(dic, output_file)
        output_file.write("\n")

    optimization = (max(list_cv, key=lambda x: x['score']))
    print(optimization)
    cv_file = open(f"./optimal_tfs_cv/{classification_algo}/{dataset_name}_optimal_tfs_cv.pkl", "wb")
    pickle.dump(optimization, cv_file)
    cv_file.close()


def tmfg_test_pipeline(data_dictionary, dataset_name, test_mode, classification_algo):
    X_train = data_dictionary['X_train']
    y_train = data_dictionary['y_train']

    data = pd.DataFrame(X_train).fillna(method="ffill").fillna(method="bfill")
    data = data.loc[:, data.std() > 0.0]
    data = data.to_numpy()
    X_train = data

    X_test = data_dictionary['X_test']
    y_test = data_dictionary['y_test']

    if test_mode == 'local':

        df = pd.read_json(f'./full_tfs_cv/{classification_algo}/{dataset_name}_tmfg_full_cv.json', lines=True)

        n_features_list = [10, 50, 100, 150, 200]
        matrix_report_dict = {}

        for i in n_features_list:

            local_df = df[df.num_features == i]
            local_df.reset_index(drop=True, inplace=True)
            optimal_values = local_df.iloc[local_df['score'].argmax()]

            np.random.seed(0)
            if classification_algo == 'LinearSVC':
                clf = LinearSVC(random_state=0, max_iter=50000)
            elif classification_algo == 'KNN':
                clf = KNeighborsClassifier()
            else:
                clf = DecisionTreeClassifier(random_state=0)
            pipeline = Pipeline([('tfs', TFS_class(num=i, dataset_name=dataset_name, alpha=optimal_values['alpha'], method=optimal_values['correlation_value'],
                                                   correlation_type=optimal_values['correlation_type'], step='test')), ('scaling', StandardScaler()), ('estimator', clf)])
            pipeline.fit(X_train, y_train)
            preds = pipeline.predict(X_test)

            c_matrix = confusion_matrix(y_test, preds)
            classification_report_dict = classification_report(y_test, preds, output_dict=True)

            matrix_report_dict[f'n_features_{i}'] = {'confusion_matrix': c_matrix, 'classification_report': classification_report_dict, 'preds': preds, 'y_true': y_test}

        output_file = open(f'./best_configs/{classification_algo}/{dataset_name}_tmfg_test_local.pkl', 'wb')
        pickle.dump(matrix_report_dict, output_file)
        output_file.close()


def ifs_test_pipeline(data_dictionary, dataset_name, test_mode, classification_algo):
    X_train = data_dictionary['X_train']
    y_train = data_dictionary['y_train']

    data = pd.DataFrame(X_train).fillna(method="ffill").fillna(method="bfill")
    data = data.loc[:, data.std() > 0.0]
    data = data.to_numpy()
    X_train = data

    X_test = data_dictionary['X_test']
    y_test = data_dictionary['y_test']

    if test_mode == 'local':

        df = pd.read_json(f'./full_ifs_cv/{classification_algo}/{dataset_name}_ifs_full_cv.json', lines=True)

        n_features_list = [10, 50, 100, 150, 200]
        matrix_report_dict = {}

        for i in n_features_list:

            local_df = df[df.num_features == i]
            local_df.reset_index(drop=True, inplace=True)
            optimal_values = local_df.iloc[local_df['score'].argmax()]

            np.random.seed(0)
            if classification_algo == 'LinearSVC':
                clf = LinearSVC(random_state=0, max_iter=50000)
            elif classification_algo == 'KNN':
                clf = KNeighborsClassifier()
            else:
                clf = DecisionTreeClassifier(random_state=0)
            pipeline = Pipeline(
                [('ifs', IFS_class(num=i, dataset_name=dataset_name, alpha=optimal_values['alpha'], factor=optimal_values['factor'], step='test')), ('scaling', StandardScaler()),
                 ('estimator', clf)])
            pipeline.fit(X_train, y_train)
            preds = pipeline.predict(X_test)

            c_matrix = confusion_matrix(y_test, preds)
            classification_report_dict = classification_report(y_test, preds, output_dict=True)

            matrix_report_dict[f'n_features_{i}'] = {'confusion_matrix': c_matrix, 'classification_report': classification_report_dict, 'preds': preds, 'y_true': y_test}

        output_file = open(f'./best_configs/{classification_algo}/{dataset_name}_ifs_test_local.pkl', 'wb')
        pickle.dump(matrix_report_dict, output_file)
        output_file.close()


def statistical_comparison_pipeline_optima(data_dictionary, dataset_name, stat_test_setting, classifiers, classification_algo):
    X_train = data_dictionary['X_train']
    y_train = pd.DataFrame(data_dictionary['y_train'])

    data = pd.DataFrame(X_train).fillna(method="ffill").fillna(method="bfill")
    data = data.loc[:, data.std() > 0.0]
    data = data.to_numpy()
    X_train = data

    df_ifs = pd.read_json(f'./full_ifs_cv/{classification_algo}/{dataset_name}_ifs_full_cv.json', lines=True)
    df_tmfg = pd.read_json(f'./full_tfs_cv/{classification_algo}/{dataset_name}_tmfg_full_cv.json', lines=True)

    n_features_list = [10, 50, 100, 150, 200]
    stat_test_to_be_run = ["paired_ttest_5x2cv"]
    stat_tests_ensamble = {}
    optimal_values_ifs = None
    optimal_values_tfs = None
    prep_pipeline_ifs = None
    prep_pipeline_tfs = None

    for stat_test in stat_test_to_be_run:

        stat_test_result_list = []

        for i in n_features_list:

            if 'ifs' in classifiers:

                if stat_test_setting == 'local':
                    local_df_ifs = df_ifs[df_ifs.num_features == i]
                    local_df_ifs.reset_index(drop=True, inplace=True)
                    optimal_values_ifs = local_df_ifs.iloc[local_df_ifs['score'].argmax()]

                prep_pipeline_ifs = ('ifs', IFS_class(num=i, dataset_name=dataset_name, alpha=optimal_values_ifs['alpha'], factor=optimal_values_ifs['factor'], step='cv'))

            if 'tfs' in classifiers:

                if stat_test_setting == 'local':
                    local_df_tmfg = df_tmfg[df_tmfg.num_features == i]
                    local_df_tmfg.reset_index(drop=True, inplace=True)
                    optimal_values_tfs = local_df_tmfg.iloc[local_df_tmfg['score'].argmax()]

                prep_pipeline_tfs = ('tfs', TFS_class(num=i, dataset_name=dataset_name, alpha=optimal_values_tfs['alpha'], method=optimal_values_tfs['correlation_value'],
                                                      correlation_type=optimal_values_tfs['correlation_type'], step='cv'))

            prep_pipeline_1 = prep_pipeline_tfs
            prep_pipeline_2 = prep_pipeline_ifs

            if classification_algo == 'LinearSVC':
                clf_1 = LinearSVC(random_state=0, max_iter=50000)
            elif classification_algo == 'KNN':
                clf_1 = KNeighborsClassifier()
            else:
                clf_1 = DecisionTreeClassifier(random_state=0)
            pipeline_1 = Pipeline([prep_pipeline_1, ('scaling', StandardScaler()), ('estimator', clf_1)])

            if classification_algo == 'LinearSVC':
                clf_2 = LinearSVC(random_state=0, max_iter=50000)
            elif classification_algo == 'KNN':
                clf_2 = KNeighborsClassifier()
            else:
                clf_2 = DecisionTreeClassifier(random_state=0)
            pipeline_2 = Pipeline([prep_pipeline_2, ('scaling', StandardScaler()), ('estimator', clf_2)])

            pipeline_1 = pipeline_1
            pipeline_2 = pipeline_2
            X_estimator_1 = X_train
            X_estimator_2 = X_train

            t, p, additional_metrics_estimator_1, additional_metrics_estimator_2 = paired_ttest_5x2cv(estimator1=pipeline_1,
                                                                                                      estimator2=pipeline_2,
                                                                                                      X_estimator_1=X_estimator_1,
                                                                                                      X_estimator_2=X_estimator_2,
                                                                                                      y_estimator_1=y_train,
                                                                                                      y_estimator_2=y_train,
                                                                                                      scoring='balanced_accuracy')

            stat_test_result_list.append((i, t, p, additional_metrics_estimator_1, additional_metrics_estimator_2))

        stat_tests_ensamble[stat_test] = stat_test_result_list

    output_file = None

    if ('tfs' in classifiers) and ('ifs' in classifiers):
        output_file = open(f'./statistical_test/{classification_algo}/{dataset_name}_stat_tests_tfs_ifs_{stat_test_setting}.pkl', 'wb')

    pickle.dump(stat_tests_ensamble, output_file)
    output_file.close()


if __name__ == '__main__':
    data_dictionary = {}
    description_dictionary = {}

    entire_datasets = glob.glob('./data/entire_datasets/*')
    splitted_datasets = {'GISETTE': glob.glob('./data/splitted_datasets/GISETTE/*'),
                         'MADELON': glob.glob('./data/splitted_datasets/MADELON/*')}

    for file in entire_datasets:
        filename = get_mat_file_name(file)
        X, y = read_mat_files(file)
        data_dictionary = train_test_split_files(X, y, filename, data_dictionary)

    for file in entire_datasets:
        filename = get_mat_file_name(file)
        description_dictionary = get_data_description(data_dictionary, filename, description_dictionary)

    for file in splitted_datasets:
        data_dictionary = read_data_files(file, splitted_datasets[file], data_dictionary)

    for file in splitted_datasets:
        description_dictionary = get_data_description(data_dictionary, file, description_dictionary)

    if args.stage == 'SM_COMPUTATION':
        produce_correlation_matrix(data_dictionary[args.dataset], args.dataset, args.cc_type)

    elif args.stage == 'IFS':
        ifs_pipeline(data_dictionary[args.dataset], args.dataset, args.classification_algo)

    elif args.stage == 'TFS':
        tmfg_pipeline(data_dictionary[args.dataset], args.dataset, args.classification_algo)

    elif args.stage == 'IFS_TEST':
        ifs_test_pipeline(data_dictionary[args.dataset], args.dataset, args.test_mode, args.classification_algo)

    elif args.stage == 'TFS_TEST':
        tmfg_test_pipeline(data_dictionary[args.dataset], args.dataset, args.test_mode, args.classification_algo)

    elif args.stage == 'STATISTICAL_TEST':
        stat_test_pair = args.stat_test_pair.split('_')
        statistical_comparison_pipeline_optima(data_dictionary[args.dataset], args.dataset, args.stat_test_setting, stat_test_pair, args.classification_algo)

    else:
        print(f'Stage {args.stage} does not exists.')
