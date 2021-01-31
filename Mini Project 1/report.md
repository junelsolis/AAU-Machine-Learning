# Mini Project 1 -- Report

Author: Junel R.S. Solis

## Introduction

For this mini project, data was provided regarding a direct marketing campaign performed by a banking institution. The data is to be prepared for and processed by a deep learning algorithm in order to predict whether a client would subscribe to the product in question -- a bank term deposit.

## Data Processing

A CSV file containing 41,188 rows of data was read into memory. Processing was required for the categorical and numerical features to be used.

### One-hot encoding for features

A number of features were non-numerical and needed to be transformed in order to be processed by the neural network. The columns _education_, _default_, _contact_ and _poutcome_ were transformed into one-hot columns. The rest of the categorical columns _job_, _marital_, _housing_, _loan_, _month_, and _day_of_week_ were dropped from the clean data set in order to reduce overfitting with too many features. The decision of which categorical columns to drop were also based on iterative manual training to see which columns provide higher evaluation accuracy.

### Numerical features

The numerical features were normalized using the **MinMaxScaler** provided by the **sklearn.preprocessing** library, with one exception being the _pdays_ column.

### pdays column

The _pdays_ column was transformed using the **StandardScaler** class provided in **sklearn.preprocessing**. The rationale behind this is that apart from the feature being used to count the number of days passed after the client was last contacted from a previous campaign, the number **999** was also used to denote **no previous contact**. This would have presented a problem if transformed using the **MinMaxScaler**. It was thought that using the **StandardScaler** would make the values less affected by outliers, and the training data seems to support this in that higher test accuracies can be achieved when the StandardScaler is used on the _pdays_ column compared to the MinMaxScaler.

It was also attempted to change the **999** values into numpy **NaN** but this type of transform would render the loss functions numberless during training.

### Encoding the labels

### Histograms

As an aid for previewing data, histograms for the cleaned numeric data were created.

The data labels are stored in the final column called "**y**". The **OrdinalEncoder** class from **sklearn.preprocessing** was used to encode the **yes** / **no** strings to **0** / **1** integers

## Modelling

## Conclusion
