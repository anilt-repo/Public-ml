import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates
from st_social_media_links import SocialMediaIcons

# Function to preprocess the data
def preprocess_data(df, date_column='Date'):
    """
    Preprocesses the data, handling different date frequencies.

    Args:
        df: Pandas DataFrame with 'Date' column and numerical columns.
        date_column: Name of the date column (default: 'Date').

    Returns:
        DataFrame with 'Date' as index and correctly formatted.  Returns None if
        the input DataFrame is invalid or an error occurs during preprocessing.
    """
    try:
        # Ensure 'Date' column exists and is the first column
        if date_column not in df.columns or df.columns[0] != date_column:
            st.error(f"Error: '{date_column}' column not found as the first column in the DataFrame.")
            return None

        # Convert 'Date' column to datetime
        df[date_column] = pd.to_datetime(df[date_column])

        # Set 'Date' as index
        df = df.set_index(date_column)

        # Ensure all remaining columns are numeric
        numeric_cols = df.select_dtypes(include=np.number).columns
        non_numeric_cols = df.columns.difference(numeric_cols)
        if len(non_numeric_cols) > 0:
            st.error(f"Error: Non-numeric columns found: {', '.join(non_numeric_cols)}. Please ensure all columns other than '{date_column}' contain only numbers.")
            return None

        if len(numeric_cols) == 0:
            st.error("Error: No numeric columns found in the DataFrame.")
            return None


        return df  # DataFrame with date index and numeric columns
    except Exception as e:
        st.error(f"An error occurred during data preprocessing: {e}")
        return None


# Function to train the model and make predictions
def forecast(df, n_periods, model_type='linear_regression'):
    """
    Trains a linear regression model on the historical data and forecasts future values.

    Args:
        df: Pandas DataFrame with date index and numerical columns.
        n_periods: Number of periods to forecast.
        model_type: Type of model to use (default: 'linear_regression').  Currently only supports linear regression.

    Returns:
        DataFrame with forecasted values. Returns None if training fails.
    """
    try:
        # Create a sequence of numbers for time
        df['time_index'] = range(len(df))

        # Prepare data for training.  Use all numeric columns as features.
        X = df[['time_index']]  # Only use time_index as predictor
        y = df.drop('time_index', axis=1) # Target is all numeric columns

        # Train the model (no need for train/test split as we use all data for training for forecasting)
        model = LinearRegression()
        model.fit(X, y)

        # Generate future time index
        future_index = np.array(range(len(df), len(df) + n_periods)).reshape(-1, 1)

        # Make predictions
        forecast_values = model.predict(future_index)
        forecast_df = pd.DataFrame(forecast_values, columns=y.columns)

        # Create future date index
        last_date = df.index[-1]
        date_frequency = pd.infer_freq(df.index)  # Infer date frequency.  This is crucial.

        if date_frequency is None:
            st.error("Error: Could not infer date frequency.  Please ensure your date data has a consistent frequency (e.g., daily, weekly, monthly).")
            return None

        future_dates = pd.date_range(start=last_date, periods=n_periods + 1, freq=date_frequency)[1:] #Skip the first date as its already present
        forecast_df.index = future_dates

        # Add the total count column
        forecast_df['Total'] = forecast_df.sum(axis=1)


        return forecast_df

    except Exception as e:
        st.error(f"An error occurred during forecasting: {e}")
        return None


# Function to plot historical and forecast data
def plot_forecast(history_df, forecast_df, chart_type='bar'):
    """
    Plots historical and forecast data using matplotlib.

    Args:
        history_df: Pandas DataFrame of historical data.
        forecast_df: Pandas DataFrame of forecasted data.
        chart_type: Type of chart to plot ('line' or 'bar').
    """

    try:
        plt.figure(figsize=(12, 6))
        if chart_type == 'line':
            for column in history_df.columns:
                plt.plot(history_df.index, history_df[column], label=f'Historical - {column}')
                plt.plot(forecast_df.index, forecast_df[column], label=f'Forecast - {column}')
        elif chart_type == 'bar':
            # Stacked bar chart
            x_hist = np.arange(len(history_df.index))
            x_forecast = np.arange(len(forecast_df.index)) + len(history_df.index)

            # Define a list of colors.  Make sure it has enough colors for all columns.
            colors = plt.cm.get_cmap('tab20').colors # or any other colormap

            # Calculate totals for historical and forecast data
            historical_totals = history_df.sum(axis=1)
            forecast_totals = forecast_df.drop('Total', axis=1).sum(axis=1) # Exclude 'Total' col, it is already sum of all columns


            # Plot historical data
            bottom = np.zeros(len(history_df.index))
            for i, column in enumerate(history_df.columns):
                color = colors[i % len(colors)]  # Cycle through colors if needed
                bars = plt.bar(x_hist, history_df[column], bottom=bottom, label=f'Historical - {column}', color=color)
                bottom += history_df[column]

                # Add numbers inside bars
                for bar in bars:
                    yval = bar.get_height()
                    if yval > 0.1: # Only show text if the segment is large enough
                        plt.text(bar.get_x() + bar.get_width()/2, bar.get_y() + yval/2, int(round(yval, 0)), ha='center', va='center', color='white', fontsize=8) #Center vertically too

            # Plot forecast data
            bottom_forecast = np.zeros(len(forecast_df.index))  # Use separate 'bottom' for forecast
            for i, column in enumerate(forecast_df.columns):
                if column != 'Total': # Don't include 'Total' in stacked bars.
                    color = colors[i % len(colors)]  # Cycle through colors if needed
                    bars = plt.bar(x_forecast, forecast_df[column], bottom=bottom_forecast, label=f'Forecast - {column}', color=color)
                    bottom_forecast += forecast_df[column]

                    # Add numbers inside bars
                    for bar in bars:
                        yval = bar.get_height()
                        if yval > 0.1: # Only show text if the segment is large enough
                            plt.text(bar.get_x() + bar.get_width()/2, bar.get_y() + yval/2, int(round(yval, 0)), ha='center', va='center', color='white', fontsize=8) #Center vertically too


            # Add total labels on top of the historical bars
            for i, total in enumerate(historical_totals):
                plt.text(x_hist[i], np.sum(history_df.iloc[i].values), int(round(total, 0)), ha='center', va='bottom', color='black', fontsize=10)

            # Add total labels on top of the forecast bars
            for i, total in enumerate(forecast_totals):
                plt.text(x_forecast[i], np.sum(forecast_df.drop('Total', axis=1).iloc[i].values), int(round(total, 0)), ha='center', va='bottom', color='black', fontsize=10)


            # Format x-axis to show only the date
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())

            plt.xticks(np.concatenate([x_hist, x_forecast]), list(history_df.index.strftime('%Y-%m-%d')) + list(forecast_df.index.strftime('%Y-%m-%d')), rotation=45) #Label with dates



        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.title('Historical and Forecast Data')
        plt.legend()
        plt.grid(True)
        plt.tight_layout() #Prevent labels from overlapping
        st.pyplot(plt)

    except Exception as e:
        st.error(f"An error occurred during plotting: {e}")



# Streamlit app
def main():
    st.title('Forecasting App')
    #st.sidebar.markdown("[![Title](https://content.linkedin.com/content/dam/me/business/en-us/amp/xbu/linkedin-revised-brand-guidelines/in-logo/fg/brand-inlogo-acceptable-follow-dsk-v01-1x.png/jcr:content/renditions/brand-inlogo-acceptable-follow-dsk-v01-2x.png)](https://www.linkedin.com/in/aniltiwari/)")
    with st.sidebar:
                  
        
         st.markdown("""

    **Key Features:**

    *   **Easy Data Upload:** Simply upload an Excel file with the first column containing dates and subsequent columns containing numerical data.
    *   **Date Frequency Handling:** The app automatically detects the frequency of your date data (e.g., daily, weekly, monthly).
    *   **Multiple Column Forecasting:**  Forecasts are generated considering all numerical columns in your data.
    *   **Interactive Charting:** Visualize your historical data and forecasts with interactive stacked bar or line charts.
    *   **Customizable Forecast Length:** Specify the number of future periods you want to forecast.
    *   **Clear Data Presentation:** Forecasted data is presented in a table with a 'Total' column for easy analysis.
    """)
         
         social_media_links = [
                                "https://medium.com/@tiw-anilk",
                                "https://www.linkedin.com/in/aniltiwari/"                                
                            ]

         social_media_icons = SocialMediaIcons(social_media_links)  # Create an instance
         st.markdown("<br>", unsafe_allow_html=True) # two blank lines
         social_media_icons.render(sidebar=True, justify_content="start")  # Call the method on the instance
         
         #st.write("Built by [Anil](https://www.linkedin.com/in/aniltiwari/)", unsafe_allow_html=True)
         st.write("Visit my website [techbabas](https://techbabas.com/)", unsafe_allow_html=True)
    
      # Brief Introduction
    st.markdown("""
    **Welcome to the Forecasting App!** 

    This app allows you to upload historical data from an Excel file and generate forecasts for future periods. It uses a simple linear regression model to predict the trends in your data.

    **Get Started:**

    1.  Upload your Excel file.
    2.  Enter the number of periods to forecast.
    3.  Select your preferred chart type.
    4.  View the forecasted data and interactive chart!
    """)

    # File upload
    uploaded_file = st.file_uploader("Upload an Excel file", type=['xlsx', 'xls'])

    if uploaded_file is not None:
        try:
            df = pd.read_excel(uploaded_file)

            # Number of periods to forecast
            n_periods = st.number_input("Enter the number of periods to forecast:", min_value=1, value=5)

            # Chart type selection
            chart_type = st.selectbox("Select chart type:", ['line', 'bar'], index=1) # bar is default (index=1)


            # Data preprocessing
            processed_df = preprocess_data(df)

            if processed_df is not None:
                # Forecasting
                forecast_df = forecast(processed_df.copy(), n_periods) #Pass a copy to avoid modifying original df
                forecast_df = forecast_df.round(0).astype(int) # round off to zero decimal place

                if forecast_df is not None:
                    # Plotting
                    plot_forecast(processed_df, forecast_df, chart_type)

                    # Display Forecasted Data
                    st.subheader("Forecasted Data")
                    st.dataframe(forecast_df)
        except Exception as e:
            st.error(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
