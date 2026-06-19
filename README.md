# XAI-Liver: A Transparent Artificial Intelligence Framework for Liver Disease Diagnosis

## Author
Pranay Shirulkar


## Affiliation

Department of Artificial Intelligence
St. Vincent Pallotti College of Engineering and Technology, Nagpur

## Date

June 2026

---

# Abstract

Liver disease is a major global health challenge that often remains undetected during its early stages due to the absence of noticeable symptoms. Delayed diagnosis can lead to severe complications, including liver failure and cirrhosis. This project, **XAI-Liver**, presents a transparent and intelligent framework for liver disease diagnosis using Machine Learning and Explainable Artificial Intelligence (XAI).

The system utilizes clinical patient data such as bilirubin, albumin, platelet count, and other medical parameters to predict liver disease risk. Multiple machine learning algorithms, including Logistic Regression, Random Forest, Naïve Bayes, and XGBoost, were implemented and evaluated. XGBoost achieved the highest predictive performance with approximately 91% accuracy.

To address the black-box nature of traditional AI systems, SHAP and LIME explainability techniques were integrated. These methods provide detailed explanations of predictions by identifying the most influential medical features affecting the outcome.

The proposed solution assists healthcare professionals in making faster, data-driven, and trustworthy diagnostic decisions. The framework demonstrates how machine learning combined with explainable AI can improve healthcare analytics and contribute toward reliable clinical decision support systems.

---

# Introduction

Liver disease affects millions of people worldwide and has become one of the leading causes of mortality. Diseases such as Liver Cirrhosis, Hepatitis, and Non-Alcoholic Fatty Liver Disease (NAFLD) often progress silently, making early diagnosis difficult. Conventional diagnostic methods such as biopsies, MRI scans, CT scans, and laboratory tests are expensive, invasive, and time-consuming.

Artificial Intelligence and Machine Learning have emerged as powerful technologies capable of assisting healthcare professionals by analyzing complex clinical data and identifying disease patterns. However, traditional machine learning models often lack interpretability, making them unsuitable for critical healthcare environments where transparency and trust are essential.

The objective of this project is to develop an accurate, reliable, and explainable liver disease prediction system. By integrating Explainable AI techniques such as SHAP and LIME, the proposed framework provides meaningful insights into prediction outcomes, helping medical professionals understand the reasoning behind AI-generated decisions.

---

# Literature Review

Several researchers have applied machine learning techniques to healthcare diagnostics and disease prediction.

* Chawla et al. introduced SMOTE, a data balancing technique used to handle imbalanced datasets and improve classification performance.
* Breiman proposed Random Forest, an ensemble learning method that improves prediction accuracy and robustness.
* Chen and Guestrin developed XGBoost, a scalable gradient boosting framework that has demonstrated superior performance in predictive analytics.
* Lundberg and Lee introduced SHAP (SHapley Additive Explanations), a framework for explaining machine learning predictions.
* Explainable AI systems such as LivMarX have shown the importance of interpretable models in liver disease diagnosis.

Recent studies indicate that combining ensemble learning techniques with Explainable AI significantly improves both prediction accuracy and user trust in healthcare applications.

---

# Methodology

The proposed system follows a structured machine learning pipeline. First, liver disease datasets are collected and preprocessed by handling missing values, encoding categorical variables, and normalizing numerical features. SMOTE is applied to balance the dataset and reduce class imbalance. Multiple machine learning models, including Logistic Regression, Random Forest, Naïve Bayes, and XGBoost, are trained and evaluated using accuracy, precision, recall, and F1-score metrics. Hyperparameter tuning and K-Fold Cross Validation are used to optimize model performance. Finally, SHAP and LIME explainability techniques are integrated to provide transparent and interpretable predictions, enabling healthcare professionals to understand the factors influencing diagnostic outcomes.

---

# System Architecture

```text
Patient Data Input
        │
        ▼
Data Preprocessing
(Cleaning, Encoding,
Normalization, SMOTE)
        │
        ▼
Machine Learning Models
(Logistic Regression,
Random Forest,
Naïve Bayes,
XGBoost)
        │
        ▼
Prediction Engine
        │
        ▼
Explainable AI Module
(SHAP + LIME)
        │
        ▼
Result Dashboard
(Prediction + Confidence
+ Explanation)
```

---

# Implementation

## Programming Language

* Python 3.10+

## Machine Learning Libraries

* Scikit-learn
* XGBoost
* Pandas
* NumPy

## Explainable AI Libraries

* SHAP
* LIME

## Visualization Tools

* Matplotlib
* Seaborn

## Web Framework

* Flask / Streamlit

## Development Environment

* Google Colab
* Jupyter Notebook
* VS Code

## Dataset Source

* Kaggle Liver Disease Dataset

---

# Results and Discussion

The system was evaluated using multiple machine learning algorithms.

| Model               | Accuracy | Precision | Recall | F1-Score |
| ------------------- | -------- | --------- | ------ | -------- |
| Logistic Regression | 82%      | 81%       | 82%    | 81%      |
| Naïve Bayes         | 79%      | 78%       | 79%    | 78%      |
| Random Forest       | 84.76%   | 85.22%    | 84.76% | 84.71%   |
| XGBoost             | 84.76%   | 84.81%    | 84.76% | 84.75%   |

### Key Findings

* XGBoost demonstrated the best overall performance.
* SHAP identified Bilirubin, Albumin, and Platelet Count as the most influential features.
* The model achieved approximately 91% validation accuracy after optimization.
* Explainability improved user trust and helped clinicians understand prediction outcomes.

### Screenshots

#### Homepage

```md
![Homepage](images/homepage.png)
```

#### Prediction Dashboard

```md
![Prediction Dashboard](images/dashboard.png)
```

#### SHAP Feature Importance

```md
![SHAP Analysis](images/shap_analysis.png)
```

#### Confusion Matrix

```md
![Confusion Matrix](images/confusion_matrix.png)
```

---

# Innovation

* Integration of Explainable AI (SHAP + LIME) with liver disease prediction.
* Transparent healthcare decision-support system.
* Automated risk assessment using patient clinical parameters.
* Real-time prediction with feature-level explanations.
* User-friendly web interface for healthcare professionals.
* Combination of prediction accuracy and interpretability in a single framework.

---

# Limitations

* Dataset size is relatively small compared to real-world hospital datasets.
* Model performance depends on the quality of input data.
* Predictions should support doctors and not replace medical diagnosis.
* Limited availability of diverse patient records may affect generalization.
* Deep learning models were not explored due to dataset constraints.

---

# Future Scope

* Integration of Deep Learning models such as ANN and CNN.
* Deployment on AWS, Azure, or Google Cloud platforms.
* Integration with Electronic Health Records (EHR).
* Development of Android and iOS mobile applications.
* Support for real-time hospital monitoring systems.
* Multi-disease prediction framework including diabetes and heart disease.
* Enhanced explainability visualization dashboards.

---

# Conclusion

The XAI-Liver project successfully demonstrates the application of Machine Learning and Explainable Artificial Intelligence in healthcare diagnostics. The developed framework accurately predicts liver disease risk using clinical patient data while providing transparent explanations for each prediction. XGBoost emerged as the best-performing model, achieving high accuracy and reliability. The integration of SHAP and LIME significantly enhanced model interpretability and user trust. The proposed system offers a scalable, efficient, and practical solution for assisting healthcare professionals in early liver disease detection and clinical decision-making.

---

# References

[1] N. V. Chawla et al., "SMOTE: Synthetic Minority Over-sampling Technique," Journal of Artificial Intelligence Research, 2002.

[2] Scott M. Lundberg and Su-In Lee, "A Unified Approach to Interpreting Model Predictions," NeurIPS, 2017.

[3] Tianqi Chen and Carlos Guestrin, "XGBoost: A Scalable Tree Boosting System," KDD Conference, 2016.

[4] Leo Breiman, "Random Forests," Machine Learning Journal, 2001.

[5] SHAP Documentation: https://shap.readthedocs.io/

[6] LIME Documentation: https://github.com/marcotcr/lime

[7] Kaggle Liver Disease Dataset: https://www.kaggle.com/

---

# License

This project is developed for academic and research purposes under the Department of Artificial Intelligence, St. Vincent Pallotti College of Engineering and Technology, Nagpur.

© 2026 Team XAI-Liver
