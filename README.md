# Chronic Disease Comorbidity Analysis

Predictive Modeling & Lifestyle Factor Analysis for Multi-Morbidity Risk

## Project Overview

This project analyzes 330K+ health records from the Kaggle "Indicators of Heart Disease" dataset to identify lifestyle factors associated with chronic disease comorbidity patterns. Using unsupervised learning (K-Means clustering) and supervised classification models, we discovered two major comorbidity groups:

1. **Stroke & Heart Disease** — strongly associated with alcohol consumption and physical health
2. **Diabetes & Kidney Disease** — strongly associated with alcohol consumption and mental health

The goal is to develop a preventative screening tool to identify at-risk individuals early and recommend lifestyle adjustments.

## Technical Stack

- **Analysis:** Python (Pandas, NumPy, Scikit-learn)
- **Visualization:** Seaborn, Matplotlib
- **Unsupervised Learning:** K-Means Clustering
- **Supervised Learning:** Logistic Regression, Random Forest, Decision Trees
- **Model Optimization:** GridSearchCV, Class-Weight Balancing, Undersampling

## Key Analytical Steps

### 1. Exploratory Data Analysis (EDA)

- **Comorbidity Patterns:** Identified strongest disease correlations (Diabetes-Kidney Disease: r=0.15, Stroke-Heart Disease: r=0.20)
- **Demographic Trends:** Age emerges as dominant factor; condition count increases 4.5x from 18-24 (0.22 conditions) to 80+ (0.97 conditions)
- **Lifestyle Factors:** General health perception, age, and difficulty walking are strongest predictors of disease burden

### 2. K-Means Clustering (Unsupervised)

Applied K-Means to discover natural disease groupings:

- **General Clustering (5 clusters):** Identified health profiles (older with elevated risk, younger healthy, etc.)
- **Comorbidity Clustering (6 clusters):** Discovered Cluster 3 (Stroke + Heart Disease) and Cluster 4 (Diabetes + Kidney Disease)
- **Lifestyle Analysis by Cluster:** Found that alcohol consumption, smoking, and difficulty walking co-occur with specific comorbidity patterns

### 3. Supervised Classification Models

Built predictive models for three outcomes:

**A. Multi-Morbidity Risk (2+ conditions vs 0-1)**
- **Random Forest (Best Model):** 78% recall, 0.831 ROC-AUC
- Catches most at-risk individuals; useful for screening

**B. Stroke & Heart Disease (Cluster 3)**
- **Decision Tree (Best Model):** 76% recall, 0.79 ROC-AUC
- Good for diagnostic purposes; identifies at-risk individuals

**C. Diabetes & Kidney Disease (Cluster 4)**
- **Logistic Regression (Best Model):** 81% recall, highest F1-score
- Balanced precision/recall; best for actionable predictions

## Key Findings

### Strongest Predictors of Comorbidity:
1. **General Health Perception** (r = -0.35) - strongest single predictor
2. **Age Category** (r = 0.32) - moderate positive association
3. **Difficulty Walking** (r = 0.30) - strongest lifestyle risk factor
4. **Physical Health Days** (r = 0.25) - poor physical health predicts higher condition count

### Comorbidity-Specific Insights:
- **Stroke + Heart Disease:** 36.6% of stroke patients also have heart disease; strongly linked to alcohol consumption and poor physical health
- **Diabetes + Kidney Disease:** 40.3% of kidney disease patients also have diabetes; linked to alcohol consumption and mental health issues
- **Common Pattern:** Both comorbidity groups show high co-occurrence with smoking and difficulty walking

## 💡 Model Performance Summary

| Model | Task | Recall | Precision | ROC-AUC | Use Case |
|-------|------|--------|-----------|---------|----------|
| Random Forest | Multi-Morbidity | 78% | 27% | 0.831 | **Screening Tool** |
| Decision Tree | Stroke + HD | 76% | 10% | 0.790 | **Diagnosis** |
| Logistic Regression | Diabetes + KD | 81% | 20% | 0.839 | **Balanced Predictions** |

**Note:** All models prioritize recall over precision to minimize false negatives in healthcare contexts (avoiding missing at-risk patients is more important than reducing false alarms).

### Key Sections

1. **EDA & Correlation Analysis** — Understand disease patterns and lifestyle associations
2. **K-Means Clustering** — Discover natural disease groupings
3. **Supervised Classification** — Build and evaluate predictive models
4. **Model Comparison** — See which models work best for each comorbidity group

## Technical Highlights

- **Class Imbalance Handling:** Applied class-weight balancing and random undersampling
- **Hyperparameter Tuning:** Used GridSearchCV to optimize model performance
- **Feature Importance Analysis:** Identified which lifestyle factors matter most for each disease type
- **Overfitting Prevention:** Compared train/test metrics to ensure generalization

## Clinical Implications

The models identify lifestyle factors that could be targeted for intervention:

- **Reduce alcohol consumption** — strongest risk factor for both comorbidity groups
- **Improve physical/mental health** — strongly associated with reduced comorbidity risk
- **Increase physical activity** — weak but consistent protective effect
- **Address mobility issues** — difficulty walking is a strong comorbidity predictor

## Future Enhancements

1. **Screening Tool App:** Build a web interface for patients to input lifestyle data and receive risk scores
2. **Recommendation Engine:** Provide personalized lifestyle adjustment suggestions
3. **Time-Series Analysis:** Track how lifestyle changes affect disease risk over time
4. **Neural Networks:** Explore deep learning for improved predictive accuracy
5. **Subgroup Analysis:** Analyze patterns by age, gender, and socioeconomic status

## Team & Contact

**Project Team:**
- Samantha Pathe (Data Modeling, Classification Models)
- Chelsea Patterson (EDA, Clustering)
- Brandy Bergher (Clustering Analysis, Visualization)

**My Contributions:**
- Built classification models (Logistic Regression, Decision Trees, Random Forests)
- Hyperparameter tuning and model optimization via GridSearchCV
- Feature importance analysis and model evaluation
- Addressed class imbalance through class-weight balancing and undersampling

**Contact:** Samantha Pathe | [GitHub](https://github.com/pathesa) | [Email](mailto:pathesa@gmail.com)

## References

- Dataset: [Indicators of Heart Disease](https://www.kaggle.com/datasets/kamilpytlak/heart-disease-health-indicators)
