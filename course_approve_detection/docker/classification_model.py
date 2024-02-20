import plotly.express as px
from matplotlib import pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import auc, confusion_matrix, f1_score, precision_score, recall_score, roc_curve, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import os
# %% [markdown]
# # Classification ML Model NB.

# %%
print('Initializing ML classification model training...')

# %% [markdown]
# # Defining function.

# %%


def model_metrics(y_train, y_pred_train, y_pred_train_proba, y_test, y_pred_test, y_pred_test_proba):
    """
    Calculate and display various classification metrics for both training and test datasets.

    Parameters:
    - y_train (array-like): True labels for the training set.
    - y_pred_train (array-like): Predicted labels for the training set.
    - y_pred_train_proba (array-like): Predicted probabilities for each class on the training set.
    - y_test (array-like): True labels for the test set.
    - y_pred_test (array-like): Predicted labels for the test set.
    - y_pred_test_proba (array-like): Predicted probabilities for each class on the test set.

    Returns:
    - 
    """
    # ---METRICS TRAIN---
    # Confusion matrix with training data.
    conf_matrix_train = confusion_matrix(y_train, y_pred_train)

    # Extract TN, FP, FN, TP from the confusion matrix for training.
    tn_train, fp_train, fn_train, tp_train = conf_matrix_train.ravel()

    # Calculate metrics for training.
    acc_score_train = np.round(accuracy_score(y_train, y_pred_train), 4)
    rec_score_train = np.round(recall_score(y_train, y_pred_train), 4)
    prec_score_train = np.round(precision_score(y_train, y_pred_train), 4)
    spec_score_train = np.round((tn_train / (tn_train + fp_train)), 4)
    f_score_train = np.round(f1_score(y_train, y_pred_train), 4)

    fpr_log_train, tpr_log_train, thr_log_train = roc_curve(
        y_train, y_pred_train_proba[:, 1])
    auc_score_train = np.round(auc(fpr_log_train, tpr_log_train), 4)

    # ---METRICS TEST---
    conf_matrix_test = confusion_matrix(y_test, y_pred_test)

    tn_test, fp_test, fn_test, tp_test = conf_matrix_test.ravel()

    acc_score_test = np.round(accuracy_score(y_test, y_pred_test), 4)
    rec_score_test = np.round(recall_score(y_test, y_pred_test), 4)
    prec_score_test = np.round(precision_score(y_test, y_pred_test), 4)
    spec_score_test = np.round((tn_test / (tn_test + fp_test)), 4)
    f_score_test = np.round(f1_score(y_test, y_pred_test), 4)

    fpr_log_test, tpr_log_test, thr_log_test = roc_curve(
        y_test, y_pred_test_proba[:, 1])
    auc_score_test = np.round(auc(fpr_log_test, tpr_log_test), 4)

    # Create a DataFrame with two rows (training and test) and six columns for the metrics.
    data_df = [
        [acc_score_train, rec_score_train, prec_score_train,
            spec_score_train, f_score_train, auc_score_train],
        [acc_score_test, rec_score_test, prec_score_test,
            spec_score_test, f_score_test, auc_score_test]
    ]

    # Define column names.
    columns_df = ['Accuracy', 'Recall', 'Precision',
                  'Specificity', 'F1-Score', 'AUC']

    # Create DataFrame.
    data_result = pd.DataFrame(data_df, columns=columns_df, index=[
                               'Metrics Train', 'Metrics Test'])

    # Return the DataFrame with the metrics.
    print('Obtained metrics:')
    print(data_result, '\n')

# %% [markdown]
# # Loading Data


# %%
# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

path_processed_data = os.path.join(
    script_dir, 'output', 'processed', 'challenge_processed.csv')

data = pd.read_csv(path_processed_data, index_col=0)

# %%
# Droping un-useful columns.
data.drop(columns=['nota_final_materia'], inplace=True)

# %%
# Selecting features and target variable.
X = data.drop(columns='materia_aprobada')
y = data['materia_aprobada']

# Random State.
rnd_st = 13

# %%
# Loading best parameters.
best_params_path = './best_params.txt'

# Initializing empty dictionary.
best_params = {}
with open(best_params_path, 'r') as file:
    for line in file:
        param, value_str = line.strip().split(': ')

        # Try converting the value to an integer
        try:
            value = int(value_str)
        except ValueError:
            # If conversion fails, keep the value as a string
            value = value_str

        best_params[param] = value

# %% [markdown]
# # Model Training
#
# In this notebook im going to use the same hyperparameters that were selected as the best performing model, im not going to execute all full notebook.
#
# The goal in this NB is to train a model by using a container.

# %% [markdown]
# ## Dividing data into train and test.

# %%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, random_state=rnd_st, test_size=0.3)

# %%
print('Data divided into train and test...')

# %% [markdown]
# ## Normalizing data.

# %%
sc = StandardScaler()
X_train_sc = sc.fit_transform(X_train)
X_test_sc = sc.transform(X_test)

# %%
print('Data normalized...')

# %% [markdown]
# ## Balancing target class.

# %%
smote = SMOTE(random_state=rnd_st)
X_train_sc_ovrsmpl, y_train_ovrsmpl = smote.fit_resample(X_train_sc, y_train)

y_train_ovrsmpl.value_counts()

# %%
print('Target class balanced...')

# %%
print('Beginning with ML model training...')
print('Loaded best params...')
print(best_params)

# %%
model = XGBClassifier(**best_params)

model.fit(X_train_sc_ovrsmpl, y_train_ovrsmpl)

# %%
print('Finished Training model...')
print('Obtaining metrics...')

# %%
# Obtaining predicted probabilities.
y_pred_train_proba = model.predict_proba(X_train_sc_ovrsmpl)
y_pred_test_proba = model.predict_proba(X_test_sc)

# Setting custom threshold
custom_thresh = 0.98

# Applying custom threshold to convert probabilities for the positive class.
y_pred_train = (y_pred_train_proba[:, 1] > custom_thresh).astype(int)
y_pred_test = (y_pred_test_proba[:, 1] > custom_thresh).astype(int)

# Evaluating model.
data_result = model_metrics(
    y_train_ovrsmpl,
    y_pred_train,
    y_pred_train_proba,
    y_test,
    y_pred_test,
    y_pred_test_proba
)
data_result

# %%
# Obtaining predictions.
X_test['prediccion_aprobacion'] = y_pred_test
X_test['probabilidad_aprobacion'] = y_pred_test_proba[:, 1]

# %%

output_dir_predictions = os.path.join(script_dir, 'output', 'predictions')
os.makedirs(output_dir_predictions, exist_ok=True)

path_predicted_data = os.path.join(
    output_dir_predictions, 'challenge_predictions_test.csv')

X_test.to_csv(path_predicted_data, index=True)
print('Saved predictions in folder "output/predictions/"')

# Saving the trained model to a file using joblib.
output_trained_model = os.path.join(script_dir, 'output', 'model')
os.makedirs(output_trained_model, exist_ok=True)

path_trained_model = os.path.join(
    output_trained_model, 'trained_classification_model_docker.joblib')

joblib.dump(model, path_trained_model)
print('Finish ML model training execution ok.')
