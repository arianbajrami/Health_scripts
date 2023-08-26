import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
import statsmodels
import numpy as np
import datetime

from pmdarima.arima import auto_arima
from datetime import datetime, timedelta
from sklearn.impute import KNNImputer
from statsmodels.tsa.statespace.sarimax import SARIMAX

class weight_updater:
    def __init__(self, start_date:datetime):
        # Initialize date range and DataFrame
        # start_date = datetime(2023, 8, 7)
        end_date = datetime.now()
        date_range = pd.date_range(start_date, end_date, freq='D')
        
        self.df = pd.DataFrame(index=date_range, columns=['weight', 'calories'])
        self.automodel = None
        self.model = None
        self.forecast = None
    
    def load_external_df(self, file_path):
        """
        Load a DataFrame from a JSON file.

        Parameters:
            file_path (str): The path to the JSON file.

        Returns:
            pd.DataFrame: The loaded DataFrame.
        """
        try:
            self.df = pd.read_json(file_path)
            
        except Exception as e:
            print(f"An error occurred: {e}")
            return None
    
    def fill_values(self):
        for idx, row in self.df.iterrows():
            # Prompt for 'weight' if it is None
            if pd.isna(row['weight']):
                new_weight = input(f"Enter weight for {idx.strftime('%d.%m.%Y')}: ")
                if new_weight:
                    self.df.at[idx, 'weight'] = new_weight
            
            # Prompt for 'calories' if it is None
            if pd.isna(row['calories']):
                new_calories = input(f"Enter calories for {idx.strftime('%d.%m.%Y')}: ")
                if new_calories:
                    self.df.at[idx, 'calories'] = new_calories
    
    def return_df(self):
        return self.df
    
    
    def calculate_weekly_stats(self):
        
        # Ensure the data is in a numeric format
        self.df = self.df.apply(pd.to_numeric, errors='coerce')
        
        weekly_mean = self.df.resample("W").mean()
        weekly_weight = self.df.resample("W").median()
        
        return weekly_mean, weekly_weight
    
    def interpolate_missing_values(self, method='linear'):
        if method == 'linear':
            self.df.interpolate(method='linear', inplace=True)
        elif method == 'time':
            self.df.interpolate(method='time', inplace=True)
        elif method == 'polynomial':
            self.df.interpolate(method='polynomial', order=2, inplace=True)
        elif method == 'spline':
            self.df.interpolate(method='spline', order=2, inplace=True)
        elif method == 'pad':
            self.df.fillna(method='pad', inplace=True)
        elif method == 'bfill':
            self.df.fillna(method='bfill', inplace=True)
        elif method == 'knn':
            imputer = KNNImputer(n_neighbors=2)
            self.df[:] = imputer.fit_transform(self.df)
        else:
            print("Invalid method specified.")
    
    def fill_nan_values(self, method='constant', value=2000):
        """
        Fill NaN values in a DataFrame.

        Parameters:
            df (pd.DataFrame): The DataFrame to be processed.
            method (str): Method to fill NaN values ('constant', 'ffill', 'bfill', 'linear').
            value (any): The constant value to fill if method is 'constant'.

        Returns:
            pd.DataFrame: DataFrame with NaN values filled.
        """
        if method == 'constant':
            self.df = self.df.fillna(value)
        elif method == 'ffill':
            self.df = self.df.fillna(method='ffill')
        elif method == 'bfill':
            self.df = self.df.fillna(method='bfill')
        elif method == 'linear':
            self.df = self.df.interpolate(method='linear')
        else:
            print("Invalid method. Choose among 'constant', 'ffill', 'bfill', 'linear'.")
            
    
    def plot_weight_calories(self):
        
        sns.set_theme()
        
        plt.figure(figsize=(12,10))
        
        plt.subplot(2,1,1)
        
        plt.plot(self.df.index, self.df["weight"], color = "blue")
        plt.title("Weight over time")
        plt.xlabel("Time")
        plt.ylabel("Weight")
        
        plt.subplot(2,1,2)
        plt.plot(self.df.index, self.df["calories"], color = "green")
        plt.title("Calories over time")
        plt.xlabel("Time")
        plt.ylabel("Calories")
        
      
        plt.show()
    
    
    def grid_search_for_forecasting(self):
        
        print(self.df.head())
        x = np.array(self.df["calories"]).reshape(-1, 1)
        self.automodel = auto_arima(self.df["weight"], X = x)
        
        print(self.automodel.summary())
    
    
    def forecast_weight_based_on_calories(self, x:dict = {"2000":[2000,2000,2000,2000,2000]}, forecast_length = 5):
        
        self.forecast = pd.DataFrame(columns = x.keys())
        
        order = (2,0,0)
        seasonal_order = (0,0,0,0)
        self.model = SARIMAX(self.df['weight'], exog=self.df[['calories']], order=order, seasonal_order=seasonal_order)
        results = self.model.fit()
        
        for key in x.keys():
            future_calories = pd.DataFrame({"calories": x[key]}, index = pd.date_range(start=datetime.today() + timedelta(days = 1), periods=forecast_length, freq = "D"))
            print(future_calories)
            forecast = results.get_forecast(steps=forecast_length, exog=future_calories)
            print(forecast.predicted_mean)
            self.forecast[key, "mean"] = forecast.predicted_mean
            confidence_intervals = forecast.conf_int()
            self.forecast[key, "low"] =confidence_intervals.iloc[:, 0]
            self.forecast[key, "up"] =confidence_intervals.iloc[:, 1]
        #self.forecast.index = pd.date_range(start=datetime.today() + timedelta(days = 1), periods=forecast_length, freq = "D")
        
        
        return self.forecast
    
    def plot_forecast(self):
        
        unique_top_level = self.forecast.columns.get_level_values(0).unique()
        
        sns.set_theme()
        plt.figure(figsize= (8,10))
        for column in unique_top_level[:3]:
        
            plt.plot(self.forecast[column, "mean"], label = f"Calories: {column}")
        plt.legend()
        plt.show()
    