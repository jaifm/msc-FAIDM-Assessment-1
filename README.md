# Open University Learning Analytics - Data Mining Project
## MSc Data Mining Assessment 1

This repository contains the analysis for the Open University Learning Analytics dataset, focusing on predictive modeling (student retention) and clustering (behavioural segmentation), following the **CRISP-DM** methodology.

### Project Structure

#### 1. Data Preparation & EDA
*   `01_Preparation_and_EDA.ipynb`: Data loading, cleaning, and feature engineering (Intensity, Regularity, Procrastination, Breadth).

#### 2. Classification (Predictive Modeling)
*   **Goal**: Predict student outcomes (Pass/Fail/Distinction/Withdrawn).
*   `02a_Classification_Multiclass.ipynb`: Multiclass classification models.
*   `02b_Classification_Binary.ipynb`: Binary classification (At-Risk vs. Safe), optimized for Recall.
*   `02c_Classification_Comparison.ipynb`: Comparative analysis of modeling approaches.
*   **Result**: XGBoost identified as the optimal model.

#### 3. Clustering (Unsupervised Learning)
*   **Goal**: Segment students based on study behaviours.
*   `03a_Clustering_Models.ipynb`: Implementation of K-Means, Hierarchical, and DBSCAN algorithms. Includes model selection (Silhouette Score, Elbow Method) and model saving.
*   **Selected Model**: K-Means (K=3).

#### 4. Interpretation & Evaluation (New)
*   `03b_Cluster_Interpretation.ipynb`: Deep-dive analysis of the identified clusters using **SHAP** (SHapley Additive exPlanations) and behavioural profiling.
    *   **Cluster 0**: "The Mainstream Learners" (Moderate engagement).
    *   **Cluster 1**: "The Diligent High-Flyers" (High intensity, early submission).
    *   **Cluster 2**: "The Disengaged / At-Risk" (Low intensity, erratic behaviour).

### Methodologies Applied
*   **CRISP-DM**: All notebooks are structured according to Cross-Industry Standard Process for Data Mining.
*   **SMOTE**: Applied for handling class imbalance in classification.
*   **Optuna**: Used for hyperparameter tuning.
*   **SHAP**: Used for model interpretability.

### Installation & Usage

1.  **Environment Setup**:
    This project requires Python 3.9+. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
    *Note: If using Numba (via SHAP), ensure `numpy < 2.4` is installed.*

2.  **Running the Notebooks**:
    Run the notebooks in numerical order (`01` -> `02` -> `03` ) to ensure identifying features and models are correctly generated and saved to `2_Outputs/`.

### Key Outputs
*   `best_classification_model.pkl`: Optimized XGBoost classifier.
*   `best_clustering_model.pkl`: Optimized K-Means clustering model.
*   `clustering_features.pkl`: Engineered feature set for segmentation.
