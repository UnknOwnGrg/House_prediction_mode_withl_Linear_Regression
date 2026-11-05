import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib

def main():
    print("Let's Start the data cleaning and model training..")

    # Paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.abspath(
        os.path.join(base_dir, "..", "Datasets", "Bangalore  house data.csv")
    )
    artifacts_dir = os.path.join(base_dir, "artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)

    model_path = os.path.join(artifacts_dir, "model.joblib")
    metadata_path = os.path.join(artifacts_dir, "model_metadata.json")

    # 1.Load Data
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at: {dataset_path}")

    df = pd.read_csv(dataset_path)
    print(f"Initial dataset shape : {df.shape}")

    # 2. Data Cleaning
    # Drop columns not useful for a simple linear model
    df2 = df.drop(['area_type', 'society', 'balcony', 'availability'], axis='columns')

    # Handle missing values
    df3 = df2.dropna()
    print(f"Shape after dropping nulls: {df3.shape}")

    # Parse 'size' column to get BHK
    # E.g '2 BHK' -> 2, '4 Bedroom' -> 4
    df3['bhk'] = df3['size'].apply(lambda x: int(x.split(' ')[0]))
    df3 = df3.drop(['size'], axis='columns')

    # Clean 'total_sqft' column
    # Handle ranges and convert to float
    def convert_sqft_to_num(x):
        tokens = x.split('-')
        if len(tokens) == 2:
            try:
                return (float(tokens[0]) + float(tokens[1])) / 2
            except:
                return float(x)
        try:
            return float(x)
        except:
            return None


    # Copying the Dataset to be safe
    df4 = df3.copy()
    df4['total_sqft'] = df4['total_sqft'].apply(convert_sqft_to_num)
    df4 = df4.dropna(subset=['total_sqft'])
    print(f"Shape after cleaning total_sqft: {df4.shape}")

    # 3. Outlier Removal
    # Floor area per bedroom ration check(remove houses where are per bhk is < 300 sqft)
    df5  = df4[~(df4.total_sqft/df4.bhk < 300)].copy()
    print(f"Shape after area per bhk filter: {df5.shape}")

    # Price per square foot outlier removal
    # Price is in Lakhs, so price_per_sqft = (price * 100000) /total_sqft
    df5['price_per_sqft'] = (df5['price'] * 100000) / df5['total_sqft']

    def remove_pps_outliers(df_in):
        df_out = pd.DataFrame()
        for key , subdf in df_in.groupby('location'):
            m = np.mean(subdf.price_per_sqft)
            st = np.std(subdf.price_per_sqft)
            reduced_df = subdf[(subdf.price_per_sqft >= (m - st)) & (subdf.price_per_sqft <= (m + st))]
            df_out = pd.concat([df_out, reduced_df], ignore_index= True)
        return df_out

    df6 = remove_pps_outliers(df5)
    print(f"Shape after price per sqft outliers filter: {df6.shape}")

    # BHK outlier removal(price of n BHK should be higher than mean of n-1 BHK for similar area)
    def remove_bhk_outliers(df_in):
        exclude_indices = np.array([])
        for location , location_df in df_in.groupby('location'):
            bhk_stats = {}
            for bhk , bhk_df in location_df.groupby('bhk'):
                bhk_stats[bhk] = {
                    'mean': np.mean(bhk_df.price_per_sqft),
                    'std': np.std(bhk_df.price_per_sqft),
                    'count': bhk_df.shape[0]
                }
            for bhk, bhk_df in location_df.groupby('bhk'):
                stats = bhk_stats.get(bhk - 1)
                if stats and stats['count'] > 5:
                    exclude_indices = np.append(exclude_indices, bhk_df[bhk_df.price_per_sqft < stats['mean']].index.values)
        return df_in.drop(exclude_indices, axis='index')

    df7 = remove_bhk_outliers(df6)
    print(f"Shape after bhk outliers filter: {df7.shape}")

    # Bathroom outlier removal (bathrooms shouldn't exceed bhk + 2)
    df8 = df7[df7.bath < df7.bhk + 2]
    print(f"Shape after bathroom filter: {df8.shape}")

    df8.location = df8.location.apply(lambda x : x.strip())
    location_stats = df8.groupby('location')['location'].agg('count').sort_values(ascending=False)

    # Let's keep location with >= 30 data points, group the rest as 'other'
    location_stats_less_than_30 = location_stats[location_stats < 30]
    df8.location = df8.location.apply(lambda x: 'other' if x in location_stats_less_than_30 else x)

    unique_locations = sorted(list(df8.location.unique()))
    if 'other' in unique_locations:
        # If already exists then remove it
        unique_locations.remove('other')
        # Keep 'Other' at the end
        unique_locations.append('other')

    print(f"Number of unique locations after grouping: {len(unique_locations)}")

    # One-hot Encoding
    dummies = pd.get_dummies(df8.location, dtype=int)
    # Drop 'other' dummy column to avoid dummy variable trap (we will represent 'other' as all zeros in encoded location cols)
    # For robust matching in FastAPI backend, let's keep all location columns in dummies,
    # but we can omit 'other' if we want. Let's drop 'other' column.
    if 'other' in dummies.columns:
        dummies = dummies.drop(['other'], axis='columns')

    df9 = pd.concat([df8, dummies], axis='columns')
    df10 = df9.drop(['location', 'price_per_sqft'], axis='columns')

    # Modeling
    X = df10.drop(['price'], axis='columns')
    y = df10['price']

    X_train, X_test, y_train, y_test= train_test_split(X, y , test_size=0.2, random_state=10)

    # Instance of the Model
    lr_clf = LinearRegression()
    lr_clf.fit(X_train , y_train)
    joblib.dump(lr_clf, model_path)
    print(f"Model saved to {model_path}")

    # Evaluation
    score = lr_clf.score(X_test, y_test)
    y_pred = lr_clf.predict(X_test)
    mse = mean_squared_error(y_test,y_pred)

    print(f"Model R2 score: {score: .4f}")
    print(f"Model MSE: {mse: .4f}")

    # Save the Artifacts
    coefficients_dict = {}
    for col, coef in zip(X.columns, lr_clf.coef_):
        coefficients_dict[col] = float(coef)

    metadata = {
        "features" : list(X.columns),
        "locations": [loc for loc in unique_locations if loc != 'other'],
        "metrics": {
            "r2_score": float(score),
            "mse": float(mse)
        },
        "intercept": float(lr_clf.intercept_),
        "coefficients": coefficients_dict

    }

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"Metadata saved to {metadata_path}")
    print("Training Finished Successfully")


if __name__ == "__main__":
    main()
