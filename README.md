>📋  README.md for code

# My Paper Title

This repository is for the working implementation of the algorithms in "Generalized Reductions: Making any Hierarchical Clustering Fair and Balanced with Low Cost"

## Requirements

To install required python packages:

```setup
pip install -r requirements.txt
```

>📋  Download data at the following links:

census - https://archive.ics.uci.edu/ml/datasets/census+income  filename: adult.data

bank - https://archive.ics.uci.edu/ml/datasets/Bank+Marketing  filename: bank-full in bank.zip

Put these two files in the same directory as the .py files.

## Data Preprocessing

For data preprocessing, run the following files:

data_preprocess.py


>📋  In the main file, change the dataset names depending on which dataset you want to use, and output dataset names and output directory according to user preferrence.

See the file for more detailed instructions

Important functions: "load_data" for preprocessing data with two colors, "load_data_multi_color" for preprocessing data with multiple colors.

The user can also explore more settings of fairness by changing the parameter values in these two functions.

Default output filenames we use:

Two colors:

Dataset  Protected Feature  Fairness (b:r)  Output File Name

Census   Gender             1:3             adult.csv

Census   Race               1:7             adult_r.csv

Bank     Marital Status     1:2             bank.csv

Bank     Age                2:3             bank_a.csv



## Validation: two colors, cost objective

For run time test, run the following file:

test_script_run_time_springer.py

>📋  The file saves data to a chosen directory. See the main function and create the directory in the same folder as the .py file in advance. The user can 

also customize the name of this directory, but remember to also customize it later when tidying up or plotting the results.


## Result Collection

Run the following file:

data_tidy_up.py - two colors

>📋  The user will need to manually change the result directory in the main function. The code outputs the mean of every data over the 5 instances.



