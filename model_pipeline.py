from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import numpy   as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore",category=FutureWarning)
import joblib

class Predict():
        def __init__(self,values:list) -> None:
              self.value=values
        def preporcessing(self,):
            data=pd.DataFrame(self.value)
            data=data.T
            data.columns=['Store', 'DayOfWeek', 'Date', 'Customers', 'Open', 'Promo',
       'StateHoliday', 'SchoolHoliday', 'StoreType', 'Assortment',
       'CompetitionDistance', 'CompetitionOpenSinceMonth',
       'CompetitionOpenSinceYear', 'Promo2', 'Promo2SinceWeek',
       'Promo2SinceYear', 'PromoInterval']
            return data
        def feature_enginnering(self,data):
                    # Convert 'Date' column to datetime
            data['Date'] = pd.to_datetime(data['Date'])

            # Extract features from 'Date' column
            data['Year'] = data['Date'].dt.year
            data['Month'] = data['Date'].dt.month
            data['Day'] = data['Date'].dt.day
            data['WeekOfYear'] = data['Date'].dt.isocalendar().week.astype(int)
            data['DayOfWeek'] = data['Date'].dt.dayofweek
            data =data.drop(columns=['Date', ], axis=1)
            ## Try to make some Feature Engineering --> Feature Extraction --> Add the new column to the main DF
            data['SalesPerCustomer'] = np.where(data['Customers'] == 0, 0, data['Store'] / data['Customers'])

            data['SalesPerDistance'] = np.where(data['CompetitionDistance'] == 0, 0, data['Store'] / data['CompetitionDistance'])

            # Merge CompetitionOpenSinceYear and CompetitionOpenSinceMonth into 'CompetitionOpenSince'
            data['CompetitionOpenSince'] = data['CompetitionOpenSinceYear']* data['CompetitionOpenSinceMonth']

            # Merge Promo2SinceYear and Promo2SinceWeek into 'Promo2Since'
            data['Promo2Since'] = data['Promo2SinceYear']  * data['Promo2SinceWeek']

            data = data.drop(columns=['CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear'], axis=1)
            data =data.drop(columns=['Promo2SinceYear', 'Promo2SinceWeek',"Promo2SinceWeek"], axis=1)
            return data
                
        def pipline(self,data):
            # # %%
            ## Separete the columns according to type (numerical or categorical)
            ['Store', 'DayOfWeek', 'Customers', 'Open', 'Promo', 'SchoolHoliday', 'CompetitionDistance', 'Promo2', 'Year', 'Month', 'Day', 'WeekOfYear', 'SalesPerCustomer', 'SalesPerDistance', 'CompetitionOpenSince', 'Promo2Since']

    
            # Define categorical and numerical columns

            num_cols_meadin =[ 'CompetitionDistance', 'SalesPerCustomer', 'SalesPerDistance', 'CompetitionOpenSince', 'Promo2Since']
            
            for col in num_cols_meadin:
                # print(col)

                data[col]=data[col].astype(float)
            
            num_cols_mostfrq=['Store','DayOfWeek','Customers', 'Open', 'Promo', 'SchoolHoliday','Promo2', 'Year', 'Month', 'Day', 'WeekOfYear']
            
            for col in num_cols_mostfrq:
                # print(col)
                data[col]=data[col].astype(float)
            
            
            
            categ_cols_lable=['StateHoliday', 'StoreType','PromoInterval']
            categ_cols_onehot=[ 'Assortment']

            # Define the pipeline for categorical columns with One-Hot Encoding
            categ_pipeline_Onehot = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),  # Handle missing values
                ('onehot', OneHotEncoder( drop='first', handle_unknown='ignore'))  # One-hot encode categorical variables
            ])

            # Custom transformer for Label Encoding
            def label_encode(X):
                le = LabelEncoder()
                return np.array([le.fit_transform(col) for col in X.T]).T

            categ_pipeline_Label = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),  # Handle missing values
                ('label_encode', FunctionTransformer(label_encode, validate=False))  # Label encode categorical variables
            ])

            # Define the pipeline for numerical columns
            num_pipeline_median = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),  # Handle missing values
                ('scaler', StandardScaler())  # Standardize the numerical features
            ])

            # Pipeline for numerical columns with most frequent imputation
            num_pipeline_most_frequent = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),  # Impute missing values with most frequent value
                # ('scaler', StandardScaler())  # Standardize the numerical features
            ])
            
            # Build the full pipeline with ColumnTransformer
            full_pipeline = ColumnTransformer(transformers=[
                ('num_pipeline_median', num_pipeline_median, num_cols_meadin),  # Numerical pipeline
                ('num_pipeline_most_frequent', num_pipeline_most_frequent, num_cols_mostfrq),  # Categorical pipeline
                ('categ_pipeline_Onehot', categ_pipeline_Onehot, categ_cols_onehot),
                ('categ_pipeline_Label', categ_pipeline_Label, categ_cols_lable) # Pipeline for 'Assortment'

            ])
            # print(11111111)
            X_train=pd.read_csv(r'X_train.csv')
            # print(X_train.columns)
            full_pipeline.fit(X_train)
            # Transform new data
            data_prepared = full_pipeline.transform(data)


            return data_prepared
        def prectict_model(self,data):
            # Step 1: Load the model
            model = joblib.load(r'XGBRegressor_best_model.pkl')

            # Step 2: Prepare your input data
            # For example, if your model expects a 2D array of features:
            # Replace the values with your actual data

            # Step 3: Make predictions
            predictions = model.predict(data)
            return  predictions

            

values = [
    2,                 # Store
    5,                 # DayOfWeek
    "2015-07-31",      # Date
    # 6064.0,           # Sales
    625,               # Customers
    1,                 # Open
    1,                 # Promo
    0,                 # StateHoliday
    1,                 # SchoolHoliday
    "a",               # StoreType
    "a",               # Assortment
    570.0,            # CompetitionDistance
    11,                # CompetitionOpenSinceMonth
    2007,              # CompetitionOpenSinceYear
    1,                 # Promo2
    13,                # Promo2SinceWeek
    2010,              # Promo2SinceYear
    "Jan,Apr,Jul,Oct" # PromoInterval
]

if __name__== "__main__":
    model_pipline=Predict(values)
    data=model_pipline.preporcessing()
    data=model_pipline.feature_enginnering(data)
    data=model_pipline.pipline(data)
    # print(data.shape,data)
    prectict_model=model_pipline.prectict_model(data)
    print(prectict_model)