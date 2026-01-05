# Stroke Prediction Using Machine Learning

## Project Information

**Team Members:**
- Zawad Ahsan
- Abdullah Sajid Nafi

**Course:** CSE 422 FALL25 
**Date:** January 2026

## Project Overview

This project implements and compares multiple machine learning models to predict stroke risk in patients based on various health and demographic features. The goal is to identify the most accurate model for early stroke prediction, which can assist healthcare professionals in preventive care.

## Dataset

**Source:** Healthcare Dataset Stroke Data  
**Download Link:** [Google Drive](https://drive.google.com/file/d/18503AUrsLd25Vd-UgQK8IDy2ZlliKQ5g/view)

The dataset contains patient information including:
- **Demographic Data:** Age, gender
- **Health Indicators:** BMI (Body Mass Index), hypertension, heart disease status
- **Lifestyle Factors:** Average glucose level, smoking status
- **Other Attributes:** Marital status, residential information

**Dataset Statistics:**
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
All three primary models (Logistic Regression, Random Forest, Neural Network) demonstrated competitive performance with different strengths:

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

### Best Model Selection
The model selection depends on deployment context:
- **For Interpretability:** Logistic Regression (medical professionals can understand coefficient weights)
- **For Balanced Performance:** Random Forest (handles overfitting well, provides feature importance)
- **For Complex Pattern Detection:** Neural Network (best for large datasets with hidden patterns)

## Clinical Implications

In medical diagnosis, **minimizing False Negatives** (missing actual stroke cases) is critical as it directly impacts patient safety. Our models were configured with class weight balancing to address this priority and improve detection of minority class (stroke cases).

## Project Structure

```
CSE422-Stroke-Prediction-Using-Machine-Learning-main/
│
├── Healthcare Dataset Stroke Data.csv          # Dataset file
├── Stroke_Prediction_Machine_Learning_Model.ipynb    # Detailed notebook with extensive comments
├── Stroke_Prediction_Machine_Learning_Model_Clean.ipynb  # Clean notebook with concise comments
├── Stroke Prediction Machine Learning Model.docx     # Project documentation
├── README.md                                   # Project overview (this file)
└── .venv/                                      # Virtual environment
```

## Installation & Setup

### Prerequisites
- Python 3.8 or higher
- Jupyter Notebook or JupyterLab

### Required Libraries
```bash
pip install pandas numpy seaborn matplotlib scikit-learn tensorflow keras imbalanced-learn
```

### Running the Project

1. **Clone or download the project:**
   ```bash
   cd CSE422-Stroke-Prediction-Using-Machine-Learning-main
   ```

2. **Activate virtual environment (if available):**
   ```bash
   .venv\Scripts\activate  # Windows
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Open Jupyter Notebook:**
   ```bash
   jupyter notebook
   ```

5. **Run the notebook:**
   - For detailed explanation: `Stroke_Prediction_Machine_Learning_Model.ipynb`
   - For clean version: `Stroke_Prediction_Machine_Learning_Model_Clean.ipynb`

## Notebooks Overview

### Stroke_Prediction_Machine_Learning_Model.ipynb
- **Purpose:** Comprehensive notebook with extensive educational comments
- **Best For:** Learning, understanding methodology, lab demonstrations
- **Features:** Detailed explanations for every step, medical context, best practices

### Stroke_Prediction_Machine_Learning_Model_Clean.ipynb
- **Purpose:** Clean, production-ready notebook with concise comments
- **Best For:** Quick review, presentation, deployment reference
- **Features:** Professional 1-2 sentence comments, streamlined workflow

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

Potential improvements for future iterations:
- Implement SMOTE for advanced class imbalance handling
- Hyperparameter tuning using GridSearchCV or RandomizedSearchCV
- Feature importance analysis to identify key stroke risk factors
- K-fold cross-validation for more robust performance estimates
- Explainable AI (SHAP values) for model interpretability in clinical settings
- Ensemble voting combining multiple models for improved accuracy

## Contributors

This project was developed as part of CSE422 coursework by:
- **Zawad Ahsan** 
- **Abdullah Sajid Nafi** 
## License

This project is created for educational purposes as part of CSE422 coursework.

## Acknowledgments

- Dataset source: Healthcare Stroke Data
- Course: CSE 422 FALL25 - Section 17

---

**Last Updated:** January 5, 2026
