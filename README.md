# BreastCancerPredictionFromCsv


This repository contains a machine learning application for breast cancer diagnosis using various classification algorithms. The application follows a series of data preprocessing, model training, and evaluation steps to predict whether a breast tumor is benign or malignant.

## Table of Contents

1. [Libraries](#libraries)
2. [Dataset Overview](#dataset-overview)
3. [Data Preprocessing](#data-preprocessing)
4. [Exploratory Data Analysis](#exploratory-data-analysis)
5. [Model Training](#model-training)
6. [Model Testing](#model-testing)
7. [Results](#results)
8. [Conclusion](#conclusion)

## Libraries

The following Python libraries are used in this project:

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`

## Dataset Overview

The dataset used in this project is related to breast cancer diagnosis. It includes features derived from a digitized image of a fine needle aspirate (FNA) of a breast mass. These features describe the characteristics of the cell nuclei present in the image.

## Data Preprocessing

1. **Displaying Dataframe**  
   We start by loading and displaying the dataset to get an initial understanding of the data.

2. **Displaying Dataset Structure**  
   The structure of the dataset, including the data types and the first few rows, is displayed.

3. **Displaying Number of Rows and Columns**  
   The shape of the dataset is checked to understand the number of rows and columns.

4. **Displaying Summary Statistics**  
   Summary statistics of the numerical columns are provided to understand the distribution and range of the data.

5. **Dropping ID Column**  
   The `id` column is dropped as it does not contribute to the model training.

6. **Displaying Dataframe Repeatedly**  
   The dataset is displayed again after dropping unnecessary columns.

7. **Displaying Updated Summary Statistics**  
   Summary statistics are checked again for the updated dataset.

8. **Checking Missing Values**  
   Missing values in each column are checked to ensure data completeness.

9. **Converting Categorical Data to Numerical Data**  
   The target variable `diagnosis` is converted from categorical to numerical. The categories "B" (Benign) and "M" (Malignant) are converted to `diagnosis_M` with values 0 and 1, respectively.

## Exploratory Data Analysis

1. **Displaying Updated Dataset**  
   The updated dataset with numerical target variables is displayed.

2. **Control of Target Variable Classes**  
   The unique values in the target variable `diagnosis_M` are listed, and the distribution of the classes is visualized.

3. **Displaying Correlation Between Columns**  
   The correlation matrix is calculated and displayed to understand the relationships between features.

4. **Creating Heatmap with Annotations**  
   A heatmap is created to visualize the correlations between different features.

5. **Displaying Columns**  
   All columns of the dataset are displayed.

6. **Selecting Numeric Rows**  
   Numeric columns are selected for further analysis.

7. **Kernel Density Estimation Graphs**  
   Kernel Density Estimation (KDE) graphs are plotted for each numeric variable.

8. **Box Plots**  
   Box plots are drawn for each numeric variable to identify outliers.

## Model Training

The dataset is split into training and testing sets. Various machine learning models are then trained:

- Logistic Regression (`lr_reg`)
- Decision Tree (`dt`)
- Bagging Classifier (`bg`)
- Random Forest Classifier (`Rf`)
- AdaBoost Classifier (`Ada`)
- Gradient Boosting Classifier (`Gb`)
- KMeans Clustering (`KM`)
- K-Nearest Neighbors Classifier (`KNN`)

## Model Testing

Each trained model is tested on the test set:

- Predictions are made using `predict()` method.
- Accuracy and R^2 scores are calculated for each model.
  
## Results

The accuracy scores for the models are as follows:

- **Logistic Regression**: 84.92%
- **Decision Tree**: 92.39%
- **Bagging Classifier**: 95.32%
- **Random Forest Classifier**: 96.49%
- **AdaBoost Classifier**: 96.49%
- **Gradient Boosting Classifier**: 97.66%
- **KMeans Clustering**: 14.03% (Note: KMeans is not suitable for classification tasks)
- **K-Nearest Neighbors Classifier**: 96.49%

## Conclusion

The Gradient Boosting Classifier achieved the highest accuracy, followed closely by the Random Forest Classifier and AdaBoost Classifier. These models are effective in predicting whether a breast tumor is benign or malignant based on the features provided.

Further improvements can be made by fine-tuning the models and performing feature engineering to enhance predictive performance.
