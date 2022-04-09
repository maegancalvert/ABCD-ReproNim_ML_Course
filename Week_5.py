import numpy as np
import pandas as pd
import os
from tqdm import tqdm

import shap; shap.initjs
import shap.explainers

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt
import seaborn as sns

from importlib import reload
import week_5_helperFx as fx


### Question 1. What is the goal of "fairness" in machine learning? ###
### C ###

### Question 2. Can machine learning perpetuate biases and/or create health disparities? ###
### B and D are both true but B is "more" correct because biases in data ###
### is not always the primary way ###

### Question 3. Which of the following situations may perpetuate biases? ###
### All but C are correct ###
### D might not have any high stakes but if it uses language ###
### certain biases encoded into language might make classification errors ###
### such as ethnic minorities are scared of dogs or don't have the finances to care for a pet ###
### Similar to the example of Spellman and softball versus quarterback ###

### Question 4. Joy Buolamwini's Inclusive Coding discusses, ###
### B ###


### Question 5. Timnit Gebru advocates for the use of, ###
### A ###


### Coding Exercises
### For these exercises, we are going to be using synthetic data
### This data is not real and was designed to highlight potential challenges in datasets and analyses

# Number of males/females in our dataset
m_n = 10000
f_n = 500

# Store performance
tr_acc, te_acc, m_te_acc, f_te_acc = [], [], [], []

# Train a couple times (synthetic data performance can vary)
num_iter = 10

for ii in tqdm(range(num_iter)):

    # Generate synthetic data
    dat = fx.Create_MDD(m_n, f_n)
    X, y = dat.X, dat.y

    # Create splits
    kf = KFold(n_splits=5)

    # Train/eval for each fold
    for tr_idx, te_idx in kf.split(y):
        # Split train/test
        tr_x, te_x = X.iloc[tr_idx, :], X.iloc[te_idx, :]
        tr_y, te_y = y.iloc[tr_idx], y.iloc[te_idx]

        # Fit our model
        logl1 = LogisticRegression(penalty='l1', solver='liblinear')
        logl1.fit(tr_x, tr_y)

        # Get our predictions
        tr_pred = logl1.predict(tr_x)
        te_pred = logl1.predict(te_x)

        # Get accuracy
        tr_acc.append(accuracy_score(tr_y, tr_pred))
        te_acc.append(accuracy_score(te_y, te_pred))

        # Stratify test set by sex
        m_te_x = te_x[te_x['sex'] == 0]
        m_te_y = te_y[te_x['sex'] == 0]
        f_te_x = te_x[te_x['sex'] == 1]
        f_te_y = te_y[te_x['sex'] == 1]

        # Get our predictions
        te_pred_m = logl1.predict(m_te_x)
        te_pred_f = logl1.predict(f_te_x)

        # Get accuracy
        m_te_acc.append(accuracy_score(m_te_y, te_pred_m))
        f_te_acc.append(accuracy_score(f_te_y, te_pred_f))


    print(f'Train Acc: {np.mean(np.array(tr_acc)):.3f} '
          f'Test Acc: {np.mean(np.array(te_acc)):.3f} ')
    print(f'Test Acc Male: {np.mean(np.array(m_te_acc)):.3f} '
          f'Test Acc Female: {np.mean(np.array(f_te_acc)):.3f}')

### The training accuracy is 0.964 and test accuracy is 0.963 for the whole sample ###
### But for males the test accuracy is 0.977 and test accuracy for females is 0.687 ###
### We would not have caught this discrepancy without visualing this ###
### This could create disparities because the features of this model are only ###
### accurate for males. The diagnosis of depression could be missed in females ###
### if this model are strictly adhered to ###

### Range of splits question ###

# Range of splits we'll try
x = np.arange(.05, .75, .025)

# Total number of subjects
n = 2000

# Create overarching arrays to store performance
perf = np.zeros(x.shape)
m_perf = np.zeros(x.shape)
f_perf = np.zeros(x.shape)

# Loop through all ranges of x
for ind, p in enumerate(tqdm(x)):

    # Composition of dataset
    m_n = int(n * (1 - p))
    f_n = int(n * p)

    # Store performance
    te_acc, m_te_acc, f_te_acc = [], [], []

    # Train a couple times (synthetic data performance can vary)=
    for ii in range(20):

        # Generate data
        dat = fx.Create_MDD(m_n, f_n)
        X, y = dat.X, dat.y

        # Create splits
        kf = KFold(n_splits=5)

        # Train/eval for each fold
        for tr_idx, te_idx in kf.split(y):
            # Split train/test
            tr_x, te_x = X.iloc[tr_idx, :], X.iloc[te_idx, :]
            tr_y, te_y = y.iloc[tr_idx], y.iloc[te_idx]

            # Fit our model
            logl1 = LogisticRegression(penalty='l1', solver='liblinear')
            logl1.fit(tr_x, tr_y)

            # Get our predictions
            te_pred = logl1.predict(te_x)

            # Get accuracy
            te_acc.append(accuracy_score(te_y, te_pred))
            # Stratify test set by sex
            m_te_x = te_x[te_x['sex'] == 0]
            m_te_y = te_y[te_x['sex'] == 0]
            f_te_x = te_x[te_x['sex'] == 1]
            f_te_y = te_y[te_x['sex'] == 1]

            # Get our predictions
            te_pred_m = logl1.predict(m_te_x)
            te_pred_f = logl1.predict(f_te_x)

            m_te_acc.append(accuracy_score(m_te_y, te_pred_m))
            f_te_acc.append(accuracy_score(f_te_y, te_pred_f))

    # Store performance
    perf[ind] = np.mean(np.array(te_acc))
    m_perf[ind] = np.mean(np.array(m_te_acc))
    f_perf[ind] = np.mean(np.array(f_te_acc))


# Plot the results
fig, ax = plt.subplots(figsize = (12, 8))

# For each model
ax.plot(x, perf, color = 'green', label = 'Total Acc')
ax.plot(x, m_perf, color = 'red', label = 'M Acc')
ax.plot(x, f_perf, color = 'blue', label = 'F Acc')
ax.scatter(x, perf, color = 'green')
ax.scatter(x, m_perf, color = 'red')
ax.scatter(x, f_perf, color = 'blue')

ax.set_title('Underrepresentation')
ax.set_xlabel('Percent F')
ax.set_ylabel('Accuracy')
ax.legend()
plt.show()

### Exercise 2 ###
### GAD Diagnosis ###

# Generate the data from the helper fx
dataset = fx.Create_GAD()
dataset.to_dframe()

# Store performance
tr_acc, te_acc = [], []

# Split X, y
y = dataset.df['g_anxd']
X = dataset.df.drop(['g_anxd', 'x'], axis = 1)

# Create splits
kf = KFold(n_splits=5)

# Train/eval for each fold
for tr_idx, te_idx in kf.split(y):

    # Split train/test
    tr_x, te_x =  X.iloc[tr_idx, :], X.iloc[te_idx, :]
    tr_y, te_y =  y.iloc[tr_idx], y.iloc[te_idx]

    # Fit our model
    logl1 = LogisticRegression(solver='liblinear')# penalty='l1',
    logl1.fit(tr_x, tr_y)

    # Get our predictions
    tr_pred = logl1.predict(tr_x)
    te_pred = logl1.predict(te_x)

    tr_p = logl1.predict_log_proba(tr_x)
    tr_logg_odds = tr_p[:,1] - tr_p[:,0]

    # Get accuracy
    tr_acc.append(accuracy_score(tr_y, tr_pred))
    te_acc.append(accuracy_score(te_y, te_pred))

print(f'Train Acc: {np.mean(np.array(tr_acc)):.3f} Test Acc: {np.mean(np.array(te_acc)):.3f}'

### SHAP ###
### Question ###
### B, C ###

# # Create the explainer
# explainer = shap.LinearExplainer(logl1, tr_x)
# shap_values = explainer(tr_x)
#
# # Plot
# shap.plots.bar(shap_values)
# shap.plots.beeswarm(shap_values)

# Create an instance of our dataset
dataset = fx.Create_GAD()

# Add the context variables
dataset.add_context()

# Store performance
tr_acc, te_acc = [], []

# Split X,y
y = dataset.df['g_anxd']
X = dataset.df.drop(['g_anxd', 'x'], axis=1)

# Create splits
kf = KFold(n_splits=5)

# Train/eval for each fold
for tr_idx, te_idx in kf.split(y):
# Split train/test
    tr_x, te_x = X.iloc[tr_idx, :], X.iloc[te_idx, :]
tr_y, te_y = y.iloc[tr_idx], y.iloc[te_idx]

# Fit our model
logl1 = LogisticRegression(penalty='l1', solver='liblinear')
logl1.fit(tr_x, tr_y)

# Get our predictions
tr_pred = logl1.predict(tr_x)
te_pred = logl1.predict(te_x)

# Get accuracy
tr_acc.append(accuracy_score(tr_y, tr_pred))
te_acc.append(accuracy_score(te_y, te_pred))

print(f'Train Acc: {np.mean(np.array(tr_acc)):.3f} Test Acc: {np.mean(np.array(te_acc)):.3f}')

explainer = shap.LinearExplainer(logl1, tr_x)
shap_values = explainer(tr_x)

# Plot
shap.plots.bar(shap_values.abs.mean(0))
shap.plots.beeswarm(shap_values)

### Now food insecurity becomes the most important feature followed by neighborhood violence and access to care ###
### It is important to add social determinants of health because what is being predicted may ###
### actually be the explanatory variables rather than sociodemographics ###
### Also, social determinants of health places the responsibility of the prediction on ###
### on society or systems rather than the individual ###
### Neighborhood Crime abcd_pnsc01 ###
### Discrimination abcd_ydmes01 ###
### Acculturation abcd_via01 ###
### Residential history abcd_rhds01 ###
### Community Risk and Protective factors abcd_crpf01 ###
### Unmet family needs demo_fam_exp1_v2 - demo_fam_exp7_v2 ###
### Educational Conditions led_school_part_101 ###
