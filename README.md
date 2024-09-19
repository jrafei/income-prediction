# Income Prediction Project

This project is part of the AI28 Machine Learning course at the University of Technology of Compiègne (UTC). The objective is to predict whether an individual's income exceeds $50K per year based on their profile using supervised machine learning techniques. The dataset used is the **Adult** dataset from the UCI Machine Learning Repository.

## Project Structure

- **data/**: Scripts for data extraction and loading.
- **images/**: Contains saved figures and charts used in data analysis and model evaluation.
- **lib/**: Contains utility functions and libraries for the project.
- **notebooks/**: Jupyter notebooks containing the detailed execution and results of the machine learning models.
- **src/**: Core Python scripts automating data preprocessing and model training.
- **run_project.py**: Script to install required libraries and execute the pipeline.

## Steps Involved

1. **Data Exploration (EDA)**:
   - Analyzed features and visualized distributions.
   - Identified and handled missing values and outliers.
   
2. **Data Preprocessing**:
   - Categorical data encoding using one-hot encoding.
   - Standardization of numerical features.

3. **Modeling**:
   - Implemented multiple models:
     - Logistic Regression (with and without regularization).
     - K-Nearest Neighbors (KNN).
     - Decision Trees.
     - Random Forests.
     - XGBoost (ensemble model).
   - Models evaluated using precision, recall, and F1-score.

4. **Best Model**:
   - XGBoost achieved the highest performance with an F1-score of 0.69 for the target class `>50K`.

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/jrafei/income-prediction.git
   cd income-prediction
      
2. Install the necessary libraries:
   ```bash
   pip install -r requirements.txt
4. Run the project:
  ```bash
  python run_project.py
  


