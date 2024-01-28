import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def Metrics(df):
    height_mean = df['Height_cm'].mean()
    height_median = df['Height_cm'].median()
    height_std = df['Height_cm'].std()
    height_min = df['Height_cm'].min()
    height_max = df['Height_cm'].max()
    weight_mean = df['Weight_kg'].mean()
    weight_median = df['Weight_kg'].median()
    weight_std = df['Weight_kg'].std()
    weight_min = df['Weight_kg'].min()
    weight_max = df['Weight_kg'].max()
    result_table = pd.DataFrame({
        'Metric': ['Mean', 'Median', 'Standard Deviation', 'Min', 'Max'],
        'Height (cm)': [height_mean, height_median, height_std, height_min, height_max],
        'Weight (kg)': [weight_mean, weight_median, weight_std, weight_min, weight_max]
    })
    print(result_table)
    return df
def M1(df_metrics):
    random_subset = df_metrics.sample(n=min(100, len(df_metrics)), random_state=42)
    train_data, test_data = train_test_split(random_subset, test_size=0.3, random_state=42)
    X_train = train_data[['Height_cm', 'Gender']]
    y_train = train_data['Weight_kg']
    model = LinearRegression()
    model.fit(X_train, y_train)
    X_test = test_data[['Height_cm', 'Gender']]
    y_test = test_data['Weight_kg']
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print(f"Mean Squared Error: {mse}")
    print("Coefficients:", model.coef_)
    print("Intercept:", model.intercept_)

def M2(df_metrics):
   
    random_subset = df_metrics.sample(n=min(1000, len(df_metrics)), random_state=42)
    train_data, test_data = train_test_split(random_subset, test_size=0.3, random_state=42)
    X_train = train_data[['Height_cm', 'Gender']]
    y_train = train_data['Weight_kg']
    model = LinearRegression()
    model.fit(X_train, y_train)
    X_test = test_data[['Height_cm', 'Gender']]
    y_test = test_data['Weight_kg']
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print(f"Mean Squared Error: {mse}")
    print("Coefficients:", model.coef_)
    print("Intercept:", model.intercept_)


def M3(df_metrics):
   
    random_subset = df_metrics.sample(n=min(5000, len(df_metrics)), random_state=42)
    train_data, test_data = train_test_split(random_subset, test_size=0.3, random_state=42)
    X_train = train_data[['Height_cm', 'Gender']]
    y_train = train_data['Weight_kg']
    model = LinearRegression()
    model.fit(X_train, y_train)
    X_test = test_data[['Height_cm', 'Gender']]
    y_test = test_data['Weight_kg']
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print(f"Mean Squared Error: {mse}")
    print("Coefficients:", model.coef_)
    print("Intercept:", model.intercept_)
def M4(df_metrics):
   
    random_subset = df_metrics.sample(n=max(1000, len(df_metrics)), random_state=42)
    train_data, test_data = train_test_split(random_subset, test_size=0.3, random_state=42)
    X_train = train_data[['Height_cm', 'Gender']]
    y_train = train_data['Weight_kg']
    model = LinearRegression()
    model.fit(X_train, y_train)
    X_test = test_data[['Height_cm', 'Gender']]
    y_test = test_data['Weight_kg']
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print(f"Mean Squared Error: {mse}")
    print("Coefficients:", model.coef_)
    print("Intercept:", model.intercept_)

df = pd.read_csv('Height_Weight.csv')
df['Height_cm'] = df['Height'] * 2.54
df['Weight_kg'] = df['Weight'] * 0.453592
df_metrics = Metrics(df)
df_metrics['Gender'] = df_metrics['Gender'].map({'Female': 0, 'Male': 1})
print("\nM1:")
M1(df_metrics)

print("\nM2")
M2(df_metrics)


print("\nM3")
M3(df_metrics)



print("\nM4")
M4(df_metrics)