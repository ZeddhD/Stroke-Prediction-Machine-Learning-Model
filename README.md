# Stroke Prediction Using Machine Learning

## Overview

A comprehensive machine learning project that implements and compares multiple algorithms to predict stroke risk in patients based on health and demographic features. The goal is to identify the most accurate model for early stroke prediction to assist healthcare professionals in preventive care.

## Dataset

**Source:** Healthcare Dataset Stroke Data  
**Download Link:** [Google Drive](https://drive.google.com/file/d/18503AUrsLd25Vd-UgQK8IDy2ZlliKQ5g/view)

### Features
- **Demographic Data:** Age, gender
- **Health Indicators:** BMI (Body Mass Index), hypertension, heart disease status
- **Lifestyle Factors:** Average glucose level, smoking status
- **Other Attributes:** Marital status, residential information

### Dataset Statistics
- Total Samples: 5,110 patients
- Features: 12 attributes
- Target Variable: Stroke (Binary: 0 = No Stroke, 1 = Stroke)
- Class Imbalance: Significant underrepresentation of stroke cases

## Methodology

### 1. Exploratory Data Analysis (EDA)
- Dataset structure analysis (shape, data types, missing values)
- Statistical summary for numerical and categorical features
- Distribution analysis (histograms, boxplots, density plots)
- Outlier detection in BMI and glucose levels
- Correlation analysis using Pearson, Spearman, and Kendall methods
- Class imbalance identification

### 2. Data Preprocessing & Feature Engineering
- **Missing Value Handling:** Median imputation for BMI column
- **Feature Selection:** Removed irrelevant features (id, ever_married, work_type, Residence_type)
- **Encoding:** Label encoding for categorical variables (gender, smoking_status)
- **Feature Scaling:** RobustScaler for outlier-resistant normalization
- **Train-Test Split:** 70% training, 30% testing with stratification

### 3. Model Training
Implemented and compared six machine learning algorithms:

1. **Logistic Regression** - Linear classifier with balanced class weights
2. **Random Forest** - Ensemble of decision trees
3. **Decision Tree** - Single tree classifier
4. **K-Nearest Neighbors (KNN)** - Instance-based learning
5. **Naive Bayes** - Probabilistic classifier
6. **Neural Network** - Deep learning model with 2 hidden layers (512 and 256 neurons)

### 4. Model Evaluation
Performance metrics used:
- **Accuracy** - Overall correctness
- **ROC-AUC Score** - Ability to distinguish between classes
- **Precision** - Correctness of positive predictions
- **Recall** - Ability to identify all positive cases
- **F1-Score** - Harmonic mean of precision and recall
- **Confusion Matrix** - Visual representation of predictions

## Key Findings

### Data Insights
- **Class Imbalance:** Stroke cases significantly underrepresented in dataset
- **Missing Data:** BMI column had missing values (handled via median imputation)
- **Outliers:** Detected in BMI and average glucose level features
- **Important Features:** Age, hypertension, heart disease, average glucose level, and BMI showed positive correlation with stroke risk

### Model Performance

**Logistic Regression:**
- Simple, interpretable, and computationally efficient
- Strong baseline performance with class weight balancing
- Best for understanding feature importance and coefficients

**Random Forest:**
- Ensemble method with robust performance
- Handles non-linear relationships effectively
- Good balance between accuracy and complexity
- Provides feature importance metrics

**Neural Network:**
- Deep learning approach with 2 hidden layers (512 and 256 neurons)
- Capable of learning complex patterns
- Higher computational cost but competitive performance
- Best for large datasets with hidden patterns

### Model Selection Guide
- **For Interpretability:** Logistic Regression (medical professionals can understand coefficient weights)
- **For Balanced Performance:** Random Forest (handles overfitting well, provides feature importance)
- **For Complex Pattern Detection:** Neural Network (best for large datasets with hidden patterns)

## Clinical Implications

In medical diagnosis, **minimizing False Negatives** (missing actual stroke cases) is critical as it directly impacts patient safety. Models were configured with class weight balancing to address this priority and improve detection of minority class (stroke cases).

## Project Structure

```
Stroke-Prediction-Machine-Learning-Model/
│
├── Healthcare Dataset Stroke Data.csv
├── Stroke Prediction Machine Learning Model.ipynb
├── Stroke Prediction Machine Learning Model Explanation.ipynb
├── Project Report.pdf
├── README.md
└── requirements.txt
```

## Installation & Setup

### Prerequisites
- Python 3.8 or higher
- Jupyter Notebook or JupyterLab

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd Stroke-Prediction-Machine-Learning-Model
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch Jupyter Notebook:**
   ```bash
   jupyter notebook
   ```

4. **Open and run the notebook:**
   - `Stroke Prediction Machine Learning Model.ipynb` - Main analysis
   - `Stroke Prediction Machine Learning Model Explanation.ipynb` - Detailed explanations

## Technologies Used

### Data Manipulation & Analysis
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing

### Data Visualization
- **Seaborn** - Statistical data visualization
- **Matplotlib** - Creating plots and charts

### Machine Learning
- **Scikit-learn** - ML algorithms, preprocessing, and evaluation metrics
- **Imbalanced-learn** - SMOTE for handling class imbalance

### Deep Learning
- **TensorFlow/Keras** - Neural network implementation

## Results Summary

The project successfully demonstrates:
- Comprehensive data exploration and preprocessing pipeline
- Multiple ML model implementation and comparison
- Proper handling of class imbalance through stratification and class weights
- Robust evaluation using multiple metrics (accuracy, ROC-AUC, precision, recall, F1-score)
- Clinical context integration for medical decision-making

## Future Enhancements

- Implement SMOTE for advanced class imbalance handling
- Hyperparameter tuning using GridSearchCV or RandomizedSearchCV
- Feature importance analysis to identify key stroke risk factors
- K-fold cross-validation for more robust performance estimates
- Explainable AI (SHAP values) for model interpretability in clinical settings
- Ensemble voting combining multiple models for improved accuracy

## License

This project is created for educational purposes.

---

**Last Updated:** January 5, 2026
