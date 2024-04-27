import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from data_utils import (unify_date_format_year,
                        clean_summary,
                        safe_json_parse
                        )


# Function to preprocess the data
def preprocess_data(input_data):
    # Clean the summary column
    input_data['summary'] = input_data['summary'].apply(clean_summary)

    # Parse JSON column to dictionary
    input_data['freebase_id_json'] = input_data['freebase_id_json'].apply(safe_json_parse)

    # Extract genre IDs from JSON column
    def extract_genre_ids(json_str):
        return list(json_str.keys()) if isinstance(json_str, dict) else []

    input_data['genre_ids'] = input_data['freebase_id_json'].apply(extract_genre_ids)

    # Convert lists in 'genre_ids' column to strings
    input_data['genre_ids'] = input_data['genre_ids'].apply(lambda x: ','.join(x) if isinstance(x, list) else x)

    # Ensure 'date' column remains in the DataFrame after cleaning
    input_data['date'] = input_data['date'].fillna('')  # Fill missing values with ''

    # Encode categorical features using OneHotEncoder
    encoder = OneHotEncoder(handle_unknown='ignore')  # Handle unknown categories during encoding
    encoded_features = encoder.fit_transform(input_data[['genre_ids']])
    encoded_df = pd.DataFrame(encoded_features.toarray(), columns=encoder.get_feature_names_out(['genre_ids']))

    # Concatenate the encoded DataFrame with the original DataFrame
    cleaned_df = pd.concat([input_data, encoded_df], axis=1)

    return cleaned_df


# Function to train and predict based on genre
def predict_year_based_on_genre(df):
    # Clean and preprocess the data
    df_cleaned = preprocess_data(df)

    # Extract features and target variable
    X = df_cleaned.drop(columns=['date'])  # Assuming 'date' column is part of the features
    y = df_cleaned['date']

    # Handle missing values in target variable
    y = y.replace('', pd.NA).dropna().astype(int)

    # Encode genre_ids for training
    encoder = OneHotEncoder(handle_unknown='ignore')
    X_encoded = encoder.fit_transform(X[['genre_ids']])

    # Filter X_encoded and y based on valid target values
    valid_indices = y.dropna().index
    X_encoded_filtered = X_encoded[valid_indices]
    y_filtered = y.dropna()

    # Split data for training and testing
    X_train, X_test, y_train, y_test = train_test_split(X_encoded_filtered, y_filtered, test_size=0.2, random_state=42)

    # Train your model and make predictions
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Evaluate model performance
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")

    # Example prediction using genre information
    new_genre_json = '{"1": "fantasy"}'  # Example JSON for a new genre
    new_genre_ids = safe_json_parse(new_genre_json)
    new_genre_ids_str = ','.join(new_genre_ids.keys())
    new_genre_ids_bin = encoder.transform([[new_genre_ids_str]])  # Include a placeholder for the date
    new_year_pred = model.predict(new_genre_ids_bin)
    print(f"Predicted date for genre_ids {new_genre_ids}: {new_year_pred[0]}")

    return model


if __name__ == "__main__":
    # Load data from text file into DataFrame
    column_names = ["length", "freebase_id", "book_name", "author_name", "date", "freebase_id_json", "summary"]
    data = pd.read_csv('../../data/datacolab_dataset/txt_format/booksummaries.txt', sep="\t", header=None,
                       names=column_names)
    unify_date_format_year(data)

    # Call the function with the loaded DataFrame
    predict_year_based_on_genre(data)
