from tokenize import Ignore
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")


def DataPreprocessing(data):
    data.drop(['ID', 'ZIP Code'], axis=1, inplace=True)
    cols = set(data.columns)
    cols_numeric = set(['Age', 'Experience', 'Income', 'CCAvg', 'Mortgage'])
    cols_categorical = list(cols - cols_numeric)
    for x in cols_categorical:
        data[x] = data[x].astype('category')
    X = data.drop('Personal Loan', axis=1)
    feature_names = X.columns
    Y = data[['Personal Loan']].values
    X["Family"].astype('category')
    X = X.values
    return X, Y, feature_names


def spilt(X):
    n_ref = 2500
    n_test = 2500
    X_ref, X_test = X[:n_ref], X[n_ref:n_ref + n_test]
    return X_ref, X_test


def Inject_outliers(X_test):
    from alibi_detect.utils.perturbation import inject_outlier_tabular
    num_cols = [0, 1, 2, 4, 6]
    perc_outlier = 35
    data = inject_outlier_tabular(
        X_test, num_cols, perc_outlier, n_std=8., min_std=6.)
    X_threshold = data.data
    return X_threshold


def detect_drift(X_ref, X_test, feature_names):
    from alibi_detect.cd import ChiSquareDrift, TabularDrift
    category_map = {3: [3, 2, 0, 1], 5: [1, 2, 3],
                    7: [0, 1], 8: [0, 1], 9: [0, 1], 10: [0, 1]}
    categories_per_feature = {f: None for f in list(category_map.keys())}
    cd = TabularDrift(X_ref, p_val=.05,
                      categories_per_feature=categories_per_feature)
    labels = ['No!', 'Yes!']
    fpreds = cd.predict(X_test, drift_type='feature')
    result = pd.DataFrame(
        columns=["column name", "Drift detected?", "Test Type", "Test value", "p-value"])
    for f in range(cd.n_features):
        stat = 'Chi2' if f in list(categories_per_feature.keys()) else 'K-S'
        fname = feature_names[f]
        is_drift = 'yes' if fpreds['data']['is_drift'][f] else 'no'
        stat_val, p_val = fpreds['data']['distance'][f], fpreds['data']['p_val'][f]
        d = {"column name": fname, "Drift detected?": is_drift,
             "Test Type": stat, "Test value": stat_val, "p-value": p_val}
        result = result.append(d, ignore_index=True)
        #print(f'{fname} -- Drift? {labels[is_drift]} -- {stat} {stat_val:.3f} -- p-value {p_val:.3f}')
    result.to_csv("../Results/Drift_result.csv", encoding='utf-8')
# ----------------------
# Main
# ----------------------


def main():

    # ------------------------------------
    # 1. Load in data
    # ------------------------------------

    df = pd.read_csv("../Data/Data.csv")

    # ------------------------------------
    # 2. Preprocessing Data
    # ------------------------------------

    X, Y, feature_names = DataPreprocessing(df)

    # ------------------------------------
    # 3. Split data into reference data and test data
    # ------------------------------------

    X_ref, X_test = spilt(X)

    # ------------------------------------
    # 4. Injecting outliers into test data in numerical columns
    # ------------------------------------

    X_test_adv = Inject_outliers(X_test)

    # ------------------------------------
    # 4. Detecting drift and saving result to csv file
    # ------------------------------------

    detect_drift(X_ref, X_test_adv, feature_names)


if __name__ == "__main__":
    main()
