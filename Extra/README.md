# Water Potability Classification Project

This folder contains a machine learning project that predicts water potability based on various water quality metrics. The workflow replicates the process used in `LAB 1`, applied to a new dataset.

## Dataset
- **File:** `water_potability.csv`
- **Description:** Contains water quality metrics for 3276 different water bodies.
- **Target Variable:** `Potability` (1 = Potable, 0 = Not Potable)
- **Features:** 
    - `ph`: pH of water (0 to 14)
    - `Hardness`: Capacity of water to precipitate soap in mg/L
    - `Solids`: Total dissolved solids in ppm
    - `Chloramines`: Amount of Chloramines in ppm
    - `Sulfate`: Amount of Sulfates dissolved in mg/L
    - `Conductivity`: Electrical conductivity of water in μS/cm
    - `Organic_carbon`: Amount of organic carbon in ppm
    - `Trihalomethanes`: Amount of Trihalomethanes in μg/L
    - `Turbidity`: Measure of light emitting property of water in NTU

## Workflow & Process

### 1. Data Analysis & Training (`new.ipynb`)
The Jupyter notebook `new.ipynb` performs the following steps:
1.  **Data Loading:** Loads the `water_potability.csv` dataset using `pandas`.
2.  **Preprocessing:** Drops rows with missing values (`NaN`) to ensure clean data for the KNN algorithm.
3.  **Visualization:** 
    - Generates a bar chart showing the count of Potable vs. Non-Potable samples.
    - Saves this plot as `plot.png`.
4.  **Model Training:**
    - Splits the data into training and testing sets (80% train, 20% test).
    - Trains a **K-Nearest Neighbors (KNN)** classifier.
5.  **Evaluation:**
    - Calculates accuracy and generates a classification report.
    - Creates a confusion matrix to visualize model performance.
    - Saves the confusion matrix as `confusionmatrix.png`.
6.  **Model Saving:** Saves the trained KNN model to `model.joblib` for later use.

### 2. Deployment (`deploy.py`)
A Streamlit application was created to allow users to interact with the trained model.
- **File:** `deploy.py`
- **Functionality:**
    - Loads the trained `model.joblib`.
    - Provides input fields for all 9 water quality metrics.
    - Predicts whether the water is potable based on the user inputs.

## How to Run

1.  **Train the Model (Optional):**
    If you want to retrain the model or regenerate the plots/artifacts, run the notebook `new.ipynb`.

2.  **Run the App:**
    To start the prediction interface, run the following command in your terminal:
    ```bash
    streamlit run deploy.py
    ```

## Artifacts Generated
- `model.joblib`: The trained machine learning model.
- `plot.png`: Bar chart of the target variable distribution.
- `confusionmatrix.png`: Confusion matrix of the model's predictions on the test set.
