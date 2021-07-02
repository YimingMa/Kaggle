import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


def preprocessing(train_data, test_data):
    """
    This function preprocesses data in the following order:
        1. drops features whose 50% data are missing;
        2. fit missing values by medians;
        3. one-hot encodes categorical features.

    - Arguments:
        - `train_data`: the data for training, contains `X_train` and `Y_train`.
            - Type: `pandas.core.frame.DataFrame`.
        - `test_data`: the data for test, contains only `X_test`.
            - Type: `pandas.core.frame.DataFrame`.
    - Returns:
        - `X_train`: the data of features in `train_data`.
            - Type: `numpy.ndarray`.
        - `Y_train`: the data of the target variable in `train_data`.
            - Type: `numpy.ndarray`.
        - `X_test`: the data of features in `test_data`.
            - Type: `numpy.ndarray`.
    """
    # ---- >>> Step 0 >>> ----
    feature_names = [
        "Pclass",
        "Sex",
        "Age",
        "SibSp",
        "Parch",
        "Fare",
        "Cabin",
        "Embarked"
    ]

    target_name = ["Survived"]

    X_train_orig = train_data[feature_names]
    Y_train = train_data[target_name].to_numpy()

    X_test_orig = test_data[feature_names]
    # ---- <<< Step 0 <<< ----

    # ---- >>> Step 1 >>> ----
    threshold = 0.5

    # Concatenate `X_train_orig` and `X_test_orig` for now.
    X_train_test = pd.concat([X_train_orig, X_test_orig], axis=0)
    assert(len(X_train_test) == len(X_train_orig) + len(X_test_orig))

    features_to_drop = []

    # Loop through `feature_names`.
    for feature in feature_names:
        feature_data = X_train_test[feature]

        # Find the number of missing values.
        num_missing_values = sum(pd.isna(feature_data))

        if num_missing_values / len(feature_data) >= threshold:
            features_to_drop.append(feature)

    # Drop features.
    X_train = X_train_orig.drop(features_to_drop, axis=1)
    X_test = X_test_orig.drop(features_to_drop, axis=1)
    # ---- <<< Step 1 <<< ----

    # ---- >>> Step 2 >>> ----
    # Find features which still have missing values.
    features_with_missing_values = []

    for i in X_train.columns:
        if sum(pd.isna(X_train[i])) > 0:
            features_with_missing_values.append(i)

    for i in X_test.columns:
        if sum(pd.isna(X_test[i])) > 0:
            if i not in features_with_missing_values:
                features_with_missing_values.append(i)

    for feature in features_with_missing_values:
        feature_train_data = X_train[feature]
        feature_test_data = X_test[feature]

        missing_train_idx = pd.isna(feature_train_data)
        missing_test_idx = pd.isna(feature_test_data)
        nonmissing_train_idx = ~ missing_train_idx

        nonmissing_train_data = feature_train_data[nonmissing_train_idx].to_numpy(copy=True)
        nonmissing_train_data.sort()
        # Find the median.
        median = nonmissing_train_data[len(nonmissing_train_data) // 2]

        missing_train_idx = X_train.index[missing_train_idx]
        missing_test_idx = X_test.index[missing_test_idx]

        for i in missing_train_idx:
            X_train.loc[i, feature] = median
        for i in missing_test_idx:
            X_test.loc[i, feature] = median
    # ---- <<< Step 2 <<< ----

    # ---- >>> Step 3 >>> ----
    feature_names = X_train.columns
    categorical_features = [
        "Pclass",
        "Sex",
        "Embarked"
    ]
    continuous_features = list(set(feature_names) - set(categorical_features))

    # Retrieve data of categorical features.
    X_train_categorical = X_train[categorical_features]
    X_test_categorical = X_test[categorical_features]

    # One-Hot encode all categorical features.
    enc = OneHotEncoder(sparse=False)

    X_train_categorical_encoded = enc.fit_transform(X_train_categorical)
    X_test_categorical_encoded = enc.transform(X_test_categorical)

    # Retrieve data of continuous features.
    X_train_continuous = X_train[continuous_features]
    X_test_continous = X_test[continuous_features]

    # Reconstruct `X_train` & `X_test`.
    X_train = np.concatenate([X_train_categorical_encoded, X_train_continuous], axis=1)
    X_test = np.concatenate([X_test_categorical_encoded, X_test_continous], axis=1)
    assert(len(X_train) == len(Y_train))

    return X_train, Y_train, X_test
