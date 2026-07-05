# 🍽️ Restaurant Aggregate Rating Prediction & Recommendation System

## 📌 Project Architecture & Theoretical Overview
This repository contains an end-to-end Machine Learning solution designed to predict the **Aggregate Rating** of global restaurants. Restaurant success depends on a complex interplay of geography, culinary offerings, target market affordability, and operational features. 

By modeling these multi-dimensional attributes, this project implements a predictive intelligence engine that can aid business owners in understanding key rating drivers and guide recommendation algorithms based on restaurant traits.

---

## 🛠️ Feature Engineering & Data Pipeline Theory

Before feeding data into advanced tree-based ensembles, a meticulous feature transformations pipeline was constructed to translate raw dataset properties into pure mathematical matrices:

### 1. Dimensionality Selection
Attributes that carry high cardinal noise or unique identifiers (such as exact addresses, restaurant names, and specific color/text codes that leak target information) were programmatically isolated and dropped. This prevents data leakage and ensures the models learn generalized patterns rather than memorizing metadata.

### 2. Multi-Label Cuisine Decomposition
Standard categorical encoders fail when a single feature contains multiple values (e.g., `"French, Japanese, Desserts"`). To address this, a **Multi-Label Binarization** technique was employed. It splits the comma-separated strings and expands them into individual binary columns ($0$ or $1$) for each unique cuisine option, ensuring the model treats every culinary influence as an independent dimension.

### 3. Logarithmic Scaling for Skewed Features
Numerical attributes like `Votes` and `Average Cost for two` typically exhibit extreme right-skewed distributions (a few restaurants have massive traction or very high costs, while the majority cluster at lower ranges). 
* To prevent massive values from dominating the model gradient updates, a logarithmic transformation formula was implemented: 
  $$y = \log(x + 1)$$
* This compresses the range of variables, dampens outliers, and exposes linear relationships to variance-based models.

### 4. Categorical Sequence Encoding
Remaining categorical variables were passed through structural label mappings to transform non-numeric string values into unique ordinal integers, formatting the entire feature space into mathematical structures optimized for fast computing matrices.

---

## 🤖 Machine Learning Model Architecture

The regression framework evaluates the predictive capacity of ensemble learning paradigms, transitioning from traditional bagged trees to optimized gradient boosting.

### 1. Baseline: Random Forest Regressor
A bagging algorithm that builds multiple independent decision trees in parallel. Each tree is trained on a random bootstrap sample of data using a random subset of features. The final prediction averages out the outputs of all individual decision trees, substantially reducing variance and preventing overfitting.

### 2. Champion: Extreme Gradient Boosting (XGBoost)
Unlike Random Forest, XGBoost is a sequential boosting framework where each subsequent tree is built specifically to correct the residual errors made by the previous trees. It optimizes a formal objective function combining a training loss (Square Error) and a regularization term (L1/L2 weights) to control model complexity.

### 3. Mathematical Optimization via Grid Search
To uncover the absolute peak performance of the XGBoost architecture, a global hyperparameter grid search was orchestrated over a **3-Fold Cross-Validation** splitting mechanism. This process systematically evaluates combinations across multiple structural parameters:
* **`n_estimators`**: Controls the total sequence of boosting rounds (trees to grow).
* **`max_depth`**: Limits the maximum depth of an individual tree to handle complex non-linear feature interactions without creating highly customized, brittle branches.
* **`learning_rate`**: Scales the step-size contribution of each new tree, forcing the optimization path to systematically converge toward global error minima.
* **`subsample` & `colsample_bytree`**: Adds stochastic regularizations by randomly sampling rows and columns per boosting step, heavily boosting resistance to noise.

---

## 📊 Model Explainability & Interpretability

To overcome the "black-box" nature of ensemble learning, this project integrates global explainability matrices to guarantee transparency:

* **SHAP (SHapley Additive exPlanations):** Grounded in cooperative game theory, SHAP calculates the exact marginal contribution of each feature to the final aggregate rating output. The generated beeswarm visualizations demonstrate not just feature importance, but the direction of an impact—illustrating exactly how specific log-cost scales or delivery services shift predictions above or below the base expected value.
* **Residual Error Analysis:** A tracking mechanism comparing true test labels against calculated predictions. Plotting the distribution of these residuals verifies if errors gather tightly around zero, ensuring the prediction variance is balanced and unbiased across all rating spectrums.

---

## 📁 Repository Structure
```text
machinelearningmodel/               # Main portfolio repository
└── restaurant-rating-prediction/   # This project space
    ├── Dataset .csv                # Core dataset containing restaurant entries
    ├── Restaurant_recommendation.ipynb  # End-to-end training and evaluation notebook
    └── requirements.txt            # Local pipeline dependencies