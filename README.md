# Rossmann Store Sales Prediction Project

This project was part of the **Digital Egyptian Pioneers Initiative (DEPI)**, where I collaborated with my classmates to analyze historical sales data, identify key factors influencing store performance, and build an accurate model to predict future sales for Rossmann stores.

## Data Source

The dataset used in this project is publicly available on Kaggle:
[Rossmann Store Sales Dataset](https://www.kaggle.com/datasets/pratyushakar/rossmann-store-sales).

## Project Goals

The primary objectives of this project were:

- Analyzing historical sales trends.
- Identifying key factors influencing store performance.
- Developing a machine learning model to predict future sales accurately.

## Data Collection Process

We obtained the **Rossmann Store Sales dataset** from Kaggle. It contains detailed information about:

- Store sales
- Promotions
- Holidays
- Store characteristics

This dataset provided the foundation for our analysis and predictive modeling.

## Exploratory Data Analysis (EDA)

We conducted thorough data exploration to:

- Identify missing values, inconsistencies, or errors.
- Explore patterns and trends using data visualization techniques.

These steps helped us gain a deeper understanding of the factors affecting store sales and prepared us for the next phase.

## Data Cleaning

During this phase, we addressed the following issues:

- **CompetitionDistance**: Missing values were filled using median values.
- **CompetitionOpenSinceMonth & CompetitionOpenSinceYear**: Missing values were interpolated.
- **Promo2SinceWeek & Promo2SinceYear**: Nulls were replaced with `0`.
- **PromoInterval**: Missing entries were labeled as "Missing."

This ensured the dataset was ready for modeling.

## Data Visualization

We used **Power BI** to visualize important patterns and trends, including:

- Sales seasonality
- The impact of promotions
- Store-specific factors

These visualizations provided key insights that guided our analysis and model development.

## Feature Engineering

To enhance model performance, we created new features, including:

- **SalesPerCustomer**: Captures the relationship between sales and customers.
- **SalesPerDistance**: Captures the relationship between sales and competition distance.
- We also merged features such as **CompetitionOpenSinceYear** with **CompetitionOpenSinceMonth** and **Promo2SinceYear** with **Promo2SinceWeek** to simplify the dataset.

## Modeling and Algorithms

We built a machine learning pipeline and experimented with various models to predict store sales. The pipeline included feature engineering, preprocessing, and model selection. The following models were optimized using **GridSearchCV**:

- **SGDRegressor**
- **KNeighborsRegressor**
- **XGBRegressor**

## Pipeline & Imputing

The preprocessing pipeline included:

- **Numerical features**: Imputed missing values using median or most frequent strategies.
- **Categorical features**: Processed using One-Hot Encoding or Label Encoding.
- **Scaling**: Applied standard scaling to numerical features for consistent input to the models.

We used **XGBoost** as our final model after optimization, which provided accurate sales predictions.

## Model Selection & Evaluation

We evaluated the performance of each model using **3-fold cross-validation** and key metrics, such as:

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² score

The **XGBRegressor** model achieved an accuracy of **97.8%** and was selected as the best-performing model.

## Model Deployment

We deployed the final model using multiple platforms:

1. **Streamlit**: A web-based application for model inference.
2. **Power BI**: For interactive visualizations and reporting.
3. **FastAPI**: To serve the model via an API for easy integration with other applications.

The model, saved as a **PKL** file, was efficiently loaded and deployed across these platforms, providing seamless access to sales predictions.

## Conclusion

Through this project, we gained a comprehensive understanding of the key factors that drive store performance for Rossmann. We successfully built and deployed a machine learning model with high accuracy to predict future sales. The insights derived from this project can help store management optimize sales strategies, promotional activities, and inventory planning to enhance overall business growth.

## Contributors

This project was completed by a team of students as part of the **DEPI (Digital Egyptian Pioneer) program**.

- **Abd El-Rahman Eldeeb**
- **Mostafa Hagag**
- **Abd El-Rahman Tony**

---

Feel free to reach out if you have any questions or feedback!
