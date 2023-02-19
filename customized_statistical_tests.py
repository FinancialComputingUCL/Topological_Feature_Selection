import numpy as np
from scipy import stats

from sklearn.metrics import get_scorer, roc_auc_score, confusion_matrix, balanced_accuracy_score
from sklearn.model_selection import KFold, train_test_split, StratifiedShuffleSplit, StratifiedKFold, cross_val_score
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import Pipeline

def simple_ttest(X_estimator_1,
    X_estimator_2,
    y_estimator_1,
    y_estimator_2,
    estimator1,
    estimator2):

    seeds = [13, 51, 137, 24659, 347, 54, 233, 21, 3322, 222]

    full_scores_estimator_1 = []
    full_scores_estimator_2 = []

    additional_metrics_estimator_1 = {}
    additional_metrics_estimator_2 = {}

    X_estimator_1 = np.array(X_estimator_1)
    X_estimator_2 = np.array(X_estimator_2)
    y_estimator_1 = np.array(y_estimator_1)
    y_estimator_2 = np.array(y_estimator_2)

    for i_s, seed in enumerate(seeds):

        kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
        folds_generator_1 = kf.split(X_estimator_1, y_estimator_1)
        folds_generator_2 = kf.split(X_estimator_2, y_estimator_2)

        fold_number = 0
        additional_metrics_estimator_1[seed] = {}

        for train_index, test_index in folds_generator_1:
            X_train, X_test = X_estimator_1[train_index], X_estimator_1[test_index]
            y_train, y_test = y_estimator_1[train_index], y_estimator_1[test_index]
            
            estimator1.fit(X_train, y_train)
            y_preds = estimator1.predict(X_test)

            cm = confusion_matrix(y_test, y_preds)
            balanced_accuracy = balanced_accuracy_score(y_test, y_preds)

            full_scores_estimator_1.append(balanced_accuracy)
            additional_metrics_estimator_1[seed][fold_number] = {'y': y_test,
                                                                 'y_preds': y_preds,
                                                                 'confusion_matrix': cm,
                                                                 'balanced_accuracy': balanced_accuracy}
            fold_number = fold_number+1


        fold_number = 0
        additional_metrics_estimator_2[seed] = {}

        for train_index, test_index in folds_generator_2:
            X_train, X_test = X_estimator_2[train_index], X_estimator_2[test_index]
            y_train, y_test = y_estimator_2[train_index], y_estimator_2[test_index]
            
            estimator2.fit(X_train, y_train)
            y_preds = estimator2.predict(X_test)

            cm = confusion_matrix(y_test, y_preds)
            balanced_accuracy = balanced_accuracy_score(y_test, y_preds)

            full_scores_estimator_2.append(balanced_accuracy)
            additional_metrics_estimator_2[seed][fold_number] = {'y': y_test,
                                                                 'y_preds': y_preds,
                                                                 'confusion_matrix': cm,
                                                                 'balanced_accuracy': balanced_accuracy}
            fold_number = fold_number+1

    t, p = stats.ttest_ind(full_scores_estimator_1, full_scores_estimator_2)

    return t, p, additional_metrics_estimator_1, additional_metrics_estimator_2

def paired_ttest_5x2cv(X_estimator_1, 
    X_estimator_2,
    estimator1, 
    estimator2, 
    y_estimator_1, 
    y_estimator_2, 
    scoring=None, 
    random_seed=None):
    
    rng = np.random.RandomState(random_seed)

    if scoring is None:
        if estimator1._estimator_type == "classifier":
            scoring = "accuracy"
        elif estimator1._estimator_type == "regressor":
            scoring = "r2"
        else:
            raise AttributeError("Estimator must " "be a Classifier or Regressor.")

    if isinstance(scoring, str):
        scorer = get_scorer(scoring)
    else:
        scorer = scoring

    variance_sum = 0.0
    first_diff = None
    seeds_list = [13, 51, 137, 24659, 347, 54, 233, 21, 3322, 222, 768, 998, 2156, 3, 6432]
    additional_metrics_estimator_1 = {}
    additional_metrics_estimator_2 = {}

    def score_diff(X_estimator_1, 
        X_estimator_2, 
    	y_estimator_1, 
        y_estimator_2, 
    	X_test_estimator_1, 
        X_test_estimator_2,
    	y_test_estimator_1, 
        y_test_estimator_2,
        additional_metrics_estimator_1,
        additional_metrics_estimator_2,
        seed,
        side,
        fold_number):

        estimator1.fit(X_estimator_1, y_estimator_1)
        y_preds = estimator1.predict(X_test_estimator_1)

        cm = confusion_matrix(y_test_estimator_1, y_preds)
        balanced_accuracy_1 = balanced_accuracy_score(y_test_estimator_1, y_preds)

        additional_metrics_estimator_1[seed][side][fold_number] = {'y': y_test_estimator_1,
                                                             'y_preds': y_preds,
                                                             'confusion_matrix': cm,
                                                             'balanced_accuracy': balanced_accuracy_1}


        estimator2.fit(X_estimator_2, y_estimator_2)
        y_preds = estimator2.predict(X_test_estimator_2)

        cm = confusion_matrix(y_test_estimator_2, y_preds)
        balanced_accuracy_2 = balanced_accuracy_score(y_test_estimator_2, y_preds)

        additional_metrics_estimator_2[seed][side][fold_number] = {'y': y_test_estimator_2,
                                                             'y_preds': y_preds,
                                                             'confusion_matrix': cm,
                                                             'balanced_accuracy': balanced_accuracy_2}

        score_diff = balanced_accuracy_1 - balanced_accuracy_2

        return score_diff, additional_metrics_estimator_1, additional_metrics_estimator_2

    X_estimator_1 = np.array(X_estimator_1)
    X_estimator_2 = np.array(X_estimator_2)
    y_estimator_1 = np.array(y_estimator_1)
    y_estimator_2 = np.array(y_estimator_2)

    for i,z in zip(range(15), seeds_list):

        randint = z
        additional_metrics_estimator_1[randint] = {}
        additional_metrics_estimator_2[randint] = {}

        additional_metrics_estimator_1[randint][1] = {}
        additional_metrics_estimator_2[randint][1] = {}

        additional_metrics_estimator_1[randint][2] = {}
        additional_metrics_estimator_2[randint][2] = {}

        try:
            X_1, X_2, y_1, y_2 = train_test_split(X_estimator_1, y_estimator_1, test_size=0.5, random_state=randint, stratify=y_estimator_1)
        except:
            X_1, X_2, y_1, y_2 = train_test_split(X_estimator_1, y_estimator_1, test_size=0.5, random_state=randint)

        try:
            X_11, X_22, y_11, y_22 = train_test_split(X_estimator_2, y_estimator_2, test_size=0.5, random_state=randint, stratify=y_estimator_2)
        except:
            X_11, X_22, y_11, y_22 = train_test_split(X_estimator_2, y_estimator_2, test_size=0.5, random_state=randint)

        score_diff_1, additional_metrics_estimator_1, additional_metrics_estimator_2 = score_diff(X_1, X_11, y_1, y_11, X_2, X_22, y_2, y_22, additional_metrics_estimator_1, additional_metrics_estimator_2, randint, 1, i)
        score_diff_2, additional_metrics_estimator_1, additional_metrics_estimator_2 = score_diff(X_2, X_22, y_2, y_22, X_1, X_11, y_1, y_11, additional_metrics_estimator_1, additional_metrics_estimator_2, randint, 2, i)

        score_mean = (score_diff_1 + score_diff_2) / 2.0
        score_var = (score_diff_1 - score_mean) ** 2 + (score_diff_2 - score_mean) ** 2
        variance_sum += score_var
        if first_diff is None:
            first_diff = score_diff_1

    numerator = first_diff
    denominator = np.sqrt(1 / 15.0 * variance_sum)
    
    t_stat = numerator / denominator
    pvalue = stats.t.sf(np.abs(t_stat), 15) * 2.0

    return float(t_stat), float(pvalue), additional_metrics_estimator_1, additional_metrics_estimator_2