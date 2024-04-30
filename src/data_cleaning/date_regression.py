import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from src.data_cleaning.data_utility_provider import DataUtilityProvider


class DataPreprocessor:
    """
    A class to preprocess the input DataFrame.

    Attributes:
        df (pandas.DataFrame): The input DataFrame.

    Methods:
        preprocess_data(): Preprocesses the input data.
    """

    def __init__(self, df):
        """
        Initializes the DataPreprocessor object.

        Args:
            df (pandas.DataFrame): The input DataFrame.
        """
        self.df = df

    def preprocess_data(self):
        """
        Preprocesses the input data.

        Returns:
            pandas.DataFrame: The cleaned and preprocessed DataFrame.
        """
        # Clean the summary column
        self.df['summary'] = self.df['summary'].apply(DataUtilityProvider.clean_summary)

        # Parse JSON column to dictionary
        self.df['freebase_id_json'] = DataUtilityProvider(df=self.df).parse_json_column('freebase_id_json')['freebase_id_json']

        # Extract genre IDs from JSON column
        self.df['genre_ids'] = self.df['freebase_id_json'].apply(
            lambda x: list(x.keys()) if isinstance(x, dict) else [])

        # Convert lists in 'genre_ids' column to strings
        self.df['genre_ids'] = self.df['genre_ids'].apply(lambda x: ','.join(x) if isinstance(x, list) else x)

        # Ensure 'date' column remains in the DataFrame after cleaning
        self.df['date'] = self.df['date'].fillna('')  # Fill missing values with ''

        # Encode categorical features using OneHotEncoder
        encoder = OneHotEncoder(handle_unknown='ignore')
        encoded_features = encoder.fit_transform(self.df[['genre_ids']])
        encoded_df = pd.DataFrame(encoded_features.toarray(), columns=encoder.get_feature_names_out(['genre_ids']))

        # Concatenate the encoded DataFrame with the original DataFrame
        cleaned_df = pd.concat([self.df, encoded_df], axis=1)

        return cleaned_df


class BookYearPredictor:
    """
    A class to predict the publication year of a book based on its genre.

    Attributes:
        df (pandas.DataFrame): The input DataFrame.
        preprocessor (DataPreprocessor): The data preprocessor object.

    Methods:
        predict_year_based_on_genre(): Trains a linear regression model to predict the publication year based on genre.
    """

    def __init__(self, df):
        """
        Initializes the BookYearPredictor object.

        Args:
            df (pandas.DataFrame): The input DataFrame.
        """
        self.df = df
        self.preprocessor = DataPreprocessor(df)

    def predict_year_based_on_genre(self):
        """
        Trains a linear regression model to predict the publication year based on genre.

        Returns:
            sklearn.linear_model.LinearRegression: The trained linear regression model.
        """
        # Clean and preprocess the data
        df_cleaned = self.preprocessor.preprocess_data()

        # Extract features and target variable
        X = df_cleaned.drop(columns=['date'])
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
        X_train, X_test, y_train, y_test = train_test_split(X_encoded_filtered, y_filtered, test_size=0.2,
                                                            random_state=42)

        # Train the model and make predictions
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Evaluate model performance
        mse = mean_squared_error(y_test, y_pred)
        print(f"Mean Squared Error: {mse}")

        # Example prediction using genre information
        new_genre_json = '{"1": "fantasy", "2": "adventure"}'
        # new_genre_ids = self.preprocessor.safe_json_parse(new_genre_json)
        new_genre_ids = DataUtilityProvider(df=df_cleaned).parse_json_column(column_name='freebase_id_json')
        new_genre_ids_str = ','.join(new_genre_ids.keys())
        new_genre_ids_bin = encoder.transform([[new_genre_ids_str]])
        new_year_pred = model.predict(new_genre_ids_bin)
        print(f"Predicted date for genre_ids {new_genre_ids}: {new_year_pred[0]}")

        return model


def main():
    # Load data from text file into DataFrame
    column_names = ["length", "freebase_id", "book_name", "author_name", "date", "freebase_id_json", "summary"]
    data = pd.read_csv('../../data/datacolab_dataset/txt_format/booksummaries.txt', sep="\t", header=None,
                       names=column_names)

    # Unify the date format to year
    data = DataUtilityProvider(data).unify_date_format_year()

    # Create an instance of the BookYearPredictor class and train the model
    predictor = BookYearPredictor(data)
    model = predictor.predict_year_based_on_genre()


if __name__ == "__main__":
    main()
