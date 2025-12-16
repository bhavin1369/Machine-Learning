# Machine Learning Regression Model Comparison

## Project Overview
This project implements and compares multiple machine learning regression models to predict the target variable (S11) using a given dataset. The models are evaluated using RMSE and R² Score, and the best-performing model is visualized using an Actual vs Predicted plot.

## Dataset
- File Name: data.csv
- Features (X): All columns except the last column
- Target (y): Last column (S11)

## Libraries Used
- pandas
- numpy
- matplotlib
- scikit-learn

## Machine Learning Models
- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor
- K-Nearest Neighbors (KNN) Regressor

## Project Steps
1. Load dataset using pandas
2. Split data into training and testing sets (80% training, 20% testing)
3. Apply feature scaling using StandardScaler
4. Train multiple regression models
5. Evaluate models using RMSE and R² Score
6. Compare model performance using a bar graph
7. Plot Actual vs Predicted values for the best model

## Evaluation Metrics
- RMSE (Root Mean Squared Error): Measures prediction error
- R² Score: Measures goodness of fit

## Best Model
Random Forest Regressor showed the best performance based on evaluation metrics.

## Visualizations
- RMSE comparison bar chart for all models
- Actual vs Predicted scatter plot for Random Forest model

## How to Run
1. Place `data.csv` in the same directory as the Python file
2. Install required libraries:
3. Run the script:

## Output
- RMSE and R² scores printed in console
- Bar graph comparing RMSE of models
- Scatter plot of Actual vs Predicted S11 values

## Author
Bhavin Muchhala & Viraj Vaghasiya
B.Tech – Information & Communication Technology

