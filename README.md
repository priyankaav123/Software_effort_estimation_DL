# Software Effort Estimation using deep learning
This code implements a deep learning-based regression model to predict software development effort using features from a dataset. It uses an extensive EDA, feature engineering, scaling, log transformation and neural network modeling with performance visualization and evaluation.

## Dataset
1) File : costeffortdataset.csv
2) Target Variable : Effort
3) Raw Features:
   - PointsAdjust: Adjusted function points.
   - PointsNonAdjust: Unadjusted function points.
   - Length: Duration of the project.
   - Transactions: Number of transactions.
   - Entities: Number of entities.
   - Adjustment: Adjustment factor.
  
## Feature Engineering 
After performing Exploratory Data Analysis, the following derived features were introduced to improve model performance:
1) PointsPerMonth: PointsAdjust / Length -> Productivity rate
2) TransactionDensity: Transactions / Entities -> Project Complexity
3) AdjustmentRatio: Adjustment / 100 -> Normalized Adjustment factor

# Workflow Overview 
## Preprocessing 
- 'Effort' is log-transformed to reduce skewness.
- Features are scaled using 'RobustScaler' to reduce the influence of outliers.
- The data is split into training and testing sets (80/20)/

## Model Architecture
A multi-layer deep neural network is build using TensorFlow/Keras: 
1) Input Layer: Dense(128) with selu, BatchNorm, Dropout.
2) Hidden Layers: Dense(64->32), each with selu, BatchNorm, Dropout.
3) Output Layer: Dense(1) with linear activation.
4) Loss Function: Huber (robust to outliers).
5) Optimizer: Adam (learning rate = 0.0003).

## Callbacks 
- EarlyStopping: Prevents overfitting by monitoring val_loss.
- ReduceLROnPlateau: Reduces learning rate when validation loss stagnates.

## Evaluation Metrics 
- R2 score: Model's explanatory power.
- RMSE, MAE: Standard regression errors.
- MAPE: Relative error.
- Accuracy: Defined as 100 - MAPE.

## Visualizations 
- Training History: Loss and MAE curves over epochs to monitor model convergence.
- Actual vs Predicted Plot: Scatter plot comparing predicted and actual effort.
- Residual Plot: Show distribution of residuals to assess model bias.

# Dependencies 
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow

## Notes
Log-transforming skewed targets and using domain-specific feature engineering greatly improves model performance.
Deep neural networks with regularization are powerful tools for handling noisy effort estimation data. 
Visualization helps interpret model reliability and areas of improvement. 


# LICENSE 

This project is licensed under the **GNU Affero General Public License v3.0 (AGPL-3.0)**.

You are free to:

- Use, study, and modify the code
- Distribute your own versions, including for commercial use

But under the following strict conditions:

- If you make any modifications and **deploy** this project (even over a network), you **must release your full source code** under the same license.
- You **must preserve** this license and provide clear attribution.
- You **cannot make this project proprietary**, rebrand without credit, or close-source any derivative works.

© 2025 Priyankaa Vijay
