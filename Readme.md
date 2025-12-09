
# Production Quantity Prediction.
Leveraging Machine Learning Model to optimize production quantity.
This project focuses on building machine learning model that predicts production quantity from Lai Manufacturing Plc, a Tree based Model (Random Forest) and a Linear model (Bayesian) were chosen as baseline models, evaluation and comparison were done to the tuned models to get the best performing model. 
Goal: To derive the optimal predictive model between a Tree based and Linear based machine learning model on a production data of limited data entries of only a thousand.
Key models: Random Forest Regressor and Bayesian Ridge Regression (tuned using Grid Search and AdaBoost regressor as Ensemble method).
Target Variable: Production quantity from production operations in the year 2023.

2. Setup and Installation
Prerequisites
•	Python 3.8+
•	pip package installer
Environment Setup
1.	Clone the repository:
2.	git clone [https://github.com/your-username/churn-prediction.git](https://github.com/your-username/churn-prediction.git)
3.	cd churn-prediction
4.	Create and activate a virtual environment:
5.	python -m venv venv
6.	source venv/bin/activate  # On Linux/macOS
7.	.\venv\Scripts\activate   # On Windows
8.	Install dependencies:
9.	pip install -r requirements.txt
(Note: The requirements.txt file should include numpy, scikit-learn, and xgboost.)
3. Data
Data Source
The data is sourced from Lai Manufacturing Plc, it entails production operations information in the year 2023 (data credit: https://www.amdari.io/dashboard-projects/dashboard-project-paths/details/125).
Feature Engineering Highlights
The features are pre-processed to handle non-linearity and high-cardinality categorical data:
•	One Hot Encoding: Applied to nominal categorical features (supplier_id, employee_id, downtime_reasons, product_id)
•	Label Encoding: Applied to ordinal categorical features (employee_rating)
•	Cyclical Encoding: Applied to time-based features (month and day) to capture periodicity.
•	Scaling: Train and Test datasets are scaled using StandardScaler.
Feature Selection Highlights
The desired features are carefully reviewed and selected before inclusion in the models.
•	Coefficient Matrix: Using a benchmark of R>0.84 as highly correlated and R<0.1 as lowly correlated, we drop all highly and lowly correlated features.
•	Bivariate Analysis: The bivariate analysis provided insight on features with little influence on target variable. Some of these features are year, product_category, shift_information e.t.c.
•	Domain knowledge: Some features like “quantity_sold” and “quality_metrics” were deselected due to their suspected negative effect on our supposed model or over reliance on the target variable.

Modeling and Hyperparameter Tuning
The Random Forest Regression model and the Bayesian Ridge regression model were compared using the 80-20% Train/Test split with both models giving an R2 score of approximately 72% each, but the random Forest showed high overfitting while the Bayes Ridge gave no fitting issues.

Random Forest Model Result
TRAIN SCORES
MAE  38.570524999999996
RMSE  46.849598520158104
R2 score  0.9560871268701058
TEST SCORES
MAE  110.33765
RMSE  131.00368378980798
R2 score  0.7173683498294464

Bayesian Ridge Model Result
TRAIN SCORES
MAE  97.05860810020687
RMSE  116.74920371361304
R2 score  0.7272982278123876
TEST SCORES
MAE  108.54917911394183
RMSE  128.50813419950242
R2 score  0.728033749201833

Using the R2 score as metric to evaluate both models without parameter tuning, it is seen that the Bayesian Ridge appears to perform better but this is subjected to final confirmation with hyperparameter tuning on both models using gridsearchCV.
