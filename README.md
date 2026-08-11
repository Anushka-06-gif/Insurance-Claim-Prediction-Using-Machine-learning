# 🚗 Insurance Claim Prediction using Machine Learning

## 📌 Project Overview

This project focuses on predicting whether a customer is likely to make an **insurance claim** based on demographic, vehicle, policy, and customer-related attributes.

The project follows an end-to-end **Machine Learning pipeline**, including data loading, data preprocessing, exploratory data analysis, categorical encoding, model training, evaluation, and feature importance analysis.

Three classification algorithms are implemented and compared:

* 📊 Logistic Regression
* 🌳 Random Forest Classifier
* 🚀 XGBoost Classifier

---

## 🎯 Objective

The main objective is to build a machine learning classification model that can predict the `Response` variable:

* `0` → No insurance claim
* `1` → Insurance claim

This can help identify customers with a higher likelihood of responding to insurance offerings or making a claim.

---

## 📂 Dataset

The project uses the `car_insurance.csv` dataset.

### Dataset Size

* **Rows:** 381,109
* **Columns:** 12

### 📋 Features

| Feature                | Description                                 |
| ---------------------- | ------------------------------------------- |
| `id`                   | Unique customer identifier                  |
| `Gender`               | Customer gender                             |
| `Age`                  | Customer age                                |
| `Driving_License`      | Whether the customer has a driving license  |
| `Region_Code`          | Customer region                             |
| `Previously_Insured`   | Whether the customer was previously insured |
| `Vehicle_Age`          | Age of the vehicle                          |
| `Vehicle_Damage`       | Whether the vehicle has previous damage     |
| `Annual_Premium`       | Annual insurance premium                    |
| `Policy_Sales_Channel` | Insurance policy sales channel              |
| `Vintage`              | Customer association duration               |
| `Response`             | Target variable                             |

The notebook loads the dataset using Pandas and performs initial dataset inspection.

---

## 🔄 Project Workflow

```text
Dataset
   ↓
Data Loading
   ↓
Data Inspection
   ↓
Data Cleaning
   ↓
Missing Value Handling
   ↓
Categorical Encoding
   ↓
Exploratory Data Analysis
   ↓
Train-Test Split
   ↓
Model Training
   ↓
Model Evaluation
   ↓
Model Comparison
   ↓
Feature Importance Analysis
```

---

## 🧹 Data Preprocessing

The following preprocessing steps are performed:

### 1️⃣ Data Loading

The dataset is loaded using Pandas.

```python
df = pd.read_csv("car_insurance.csv")
```

### 2️⃣ Missing Value Handling

Missing numerical values are handled using median imputation.

```python
df.fillna(
    df.median(numeric_only=True).fillna(0),
    inplace=True
)
```

### 3️⃣ Column Name Standardization

Column names are converted into a consistent lowercase format.

```python
df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
)
```

### 4️⃣ Categorical Encoding

Categorical variables such as:

* Gender
* Vehicle Age
* Vehicle Damage

are converted into numerical representations using `LabelEncoder`.

```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

categorical_cols = [
    'gender',
    'vehicle_age',
    'vehicle_damage'
]

for col in categorical_cols:
    df[col] = le.fit_transform(df[col])
```

---

## 📊 Exploratory Data Analysis

The project performs exploratory analysis to understand the distribution of insurance claim responses.

### 📈 Insurance Claim Response Distribution

A count plot is created to visualize the distribution of:

* `0` → No Claim
* `1` → Claim

```python
sns.countplot(x='response', data=df)

plt.title("Insurance Claim Response Distribution")
plt.xlabel("Response (0 = No Claim, 1 = Claim)")
plt.ylabel("Count")
plt.show()
```

This helps identify the target-class distribution before model development.

---

## ✂️ Train-Test Split

The dataset is divided into training and testing sets using an **80:20 split**.

```python
X = df.drop('response', axis=1)
y = df['response']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

This creates separate datasets for model training and unbiased evaluation.

---

# 🤖 Machine Learning Models

## 1️⃣ Logistic Regression

Logistic Regression is used as a baseline classification model.

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)
```

The model is evaluated using:

* Accuracy
* Confusion Matrix
* Classification Report

---

## 2️⃣ Random Forest Classifier 🌳

A Random Forest model is trained using 100 decision trees.

```python
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train, y_train)
```

The model is evaluated using:

* Accuracy
* Confusion Matrix
* Classification Report

---

## 3️⃣ XGBoost Classifier 🚀

XGBoost is implemented as an advanced boosting-based classification model.

```python
from xgboost import XGBClassifier

xgb_model = XGBClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=6,
    random_state=42,
    eval_metric='logloss'
)

xgb_model.fit(X_train, y_train)
```

The XGBoost model is evaluated using accuracy, confusion matrix, and classification report.

---

# 📏 Model Evaluation

The project compares the performance of all three models:

| Model               | Evaluation                                        |
| ------------------- | ------------------------------------------------- |
| Logistic Regression | Accuracy, Confusion Matrix, Classification Report |
| Random Forest       | Accuracy, Confusion Matrix, Classification Report |
| XGBoost             | Accuracy, Confusion Matrix, Classification Report |

The notebook directly compares the accuracy of:

```python
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred))
print("Random Forest Accuracy:", accuracy_score(y_test, rf_pred))
print("XGBoost Accuracy:", accuracy_score(y_test, xgb_pred))
```

---

# 🌟 Feature Importance

Random Forest feature importance is calculated to identify which variables contribute most to the model's predictions.

```python
feature_importance = pd.Series(
    rf_model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print(feature_importance)
```

This provides insight into the relative importance of customer, vehicle, and policy-related attributes.

---

## 🛠️ Technologies Used

### Programming

* 🐍 Python

### Data Analysis

* Pandas
* NumPy

### Data Visualization

* Matplotlib
* Seaborn

### Machine Learning

* Scikit-learn
* Logistic Regression
* Random Forest
* XGBoost

### Development Environment

* Jupyter Notebook
* Google Colab

---

## 📦 Python Libraries

```text
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
```

Install the required libraries using:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost
```

---

## 📁 Project Structure

```text
Insurance-Claim-Prediction/
│
├── 📓 Insurance_Claim_Prediction.ipynb
├── 📊 car_insurance.csv
├── 📄 README.md
└── 📸 images/
    └── model_results.png
```

> Keep the dataset file in the same directory as the notebook, or update the dataset path in the code.

---

## 🔍 Key Project Highlights

* 📊 Performed **Exploratory Data Analysis (EDA)**
* 🧹 Applied **Data Cleaning and Preprocessing**
* 🔄 Handled missing numerical values
* 🔢 Applied **Categorical Encoding**
* 📈 Analyzed insurance claim response distribution
* ✂️ Created an **80:20 Train-Test Split**
* 🤖 Implemented multiple **Machine Learning Classification Models**
* 🌳 Performed **Random Forest Feature Importance Analysis**
* 🚀 Implemented **XGBoost Classification**
* 📏 Evaluated models using **Accuracy, Confusion Matrix, and Classification Report**
* ⚖️ Compared multiple machine learning models

---

## 💡 Business Applications

A model like this can support insurance businesses in:

* 🎯 Customer targeting
* 📢 Insurance campaign optimization
* 👥 Customer segmentation
* 📊 Policy sales analysis
* 🔎 Identifying customers with higher response likelihood
* 💼 Data-driven decision making

---

## 🚀 Future Improvements

Potential improvements for the project include:

* 🔧 Hyperparameter tuning using GridSearchCV or RandomizedSearchCV
* ⚖️ Handling class imbalance using appropriate techniques
* 📈 ROC-AUC and Precision-Recall analysis
* 🧠 Advanced feature engineering
* 🔍 SHAP-based model explainability
* 💾 Saving the best trained model using Joblib
* 🌐 Developing an interactive prediction application
* 📊 Creating a Power BI dashboard for insurance analytics

---

## 👩‍💻 Author

**Anushka Bairagi**

🎓 Information Technology Engineering
📊 Aspiring Data Analyst | Data Scientist | AI/ML Enthusiast
