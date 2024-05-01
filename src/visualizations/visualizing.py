import matplotlib.pyplot as plt
import pandas as pd

from src.data_cleaning.data_cleaning import DataCleaner
from src.data_cleaning.data_utility_provider import DataUtilityProvider
from src.data_cleaning.missing_handler import MissingHandler


class BookDataVisualizer:
    """
    A class to visualize various aspects of a book dataset.

    Attributes:
        data (pandas.DataFrame): The input DataFrame containing book data.

    Methods:
        plot_books_published_per_year(): Plots the number of books published per year.
        plot_genre_over_time(genre): Plots the occurrences of a specific genre over time.
        plot_top_genres_over_time(): Plots the top 5 popular genres over time.
        plot_top_authors(): Plots the top 5 most active authors based on published books.
        plot_books_per_genre(): Plots the number of books per genre.
    """

    def __init__(self, data):
        """
        Initializes the BookDataVisualizer object.

        Args:
            data (pandas.DataFrame): The input DataFrame containing book data.
        """
        self.data = data

    def plot_books_published_per_year(self):
        """
        Plots the number of books published per year.

        The function counts the number of books published in each year and creates a bar plot.
        The plot is saved to the '../../results/visualization/books_published_per_year.png' file and displayed.
        """
        # Count the number of books published in each year
        year_counts = self.data['date'].value_counts().sort_index()

        # Sort the result based on the year
        year_counts = year_counts.sort_index()

        # Create a bar plot
        plt.figure(figsize=(10, 6))
        plt.bar(year_counts.index, year_counts.values)
        plt.xlabel('Year')
        plt.ylabel('Number of Books Published')
        plt.title('Number of Books Published per Year')
        plt.xticks(rotation=45)
        plt.savefig('../../results/visualization/books_published_per_year.png')
        plt.show()

    def get_top_genres(self, top_n=10):
        """
        Get the top N most popular genres in the dataset.

        Parameters:
        top_n (int): The number of top genres to return (default is 5).

        Returns:
        list: A list of the top N most popular genres, or a list of default genres if no genres are found.
        """
        # Handle missing and "Unknown" genre names
        genre_data = self.data[self.data['freebase_id_json'].notna() & (self.data['freebase_id_json'] != '')]
        genre_counts = genre_data['freebase_id_json'].apply(lambda x: list(x.values())).explode().value_counts()

        # Get the top N most popular genres
        top_genres = genre_counts.head(top_n).index.tolist()

        # If there are less than top_n genres, return a default list of genres
        if len(top_genres) < top_n:
            return ['Default Genre {}'.format(i) for i in range(1, top_n + 1)]
        else:
            return top_genres

    def plot_genre_over_time(self, genre):
        """
        Plots the occurrences of a specific genre over time.

        Args:
            genre (str): The genre to be plotted.

        The function extracts the occurrences of the given genre over time and creates a line plot. The plot is saved
        to a file with the genre name in the filename (e.g., '../results/visualization/genre_name.png') and displayed.
        """
        genre_data = self.get_genre_data()

        time_periods = sorted(genre_data.keys())
        genre_occurrences = [genre_data[year][genre] for year in time_periods]

        plt.figure(figsize=(10, 6))
        plt.plot(time_periods, genre_occurrences, marker='o')
        plt.xlabel('Year')
        plt.ylabel('Number of Occurrences')
        plt.title(f'{genre} Over Time')
        plt.xticks(rotation=45)
        filename = f'../../results/visualization/{genre.replace("/", "_")}.png'
        plt.savefig(filename)
        plt.show()
        plt.close()

    def plot_top_genres_over_time(self):
        """
        Plots the top 5 popular genres over time.

        The function extracts the top 5 popular genres and creates individual plots for each genre. It also creates a
        combined plot of the top 5 genres over time and saves it to the '../results/visualization/top_genres.png' file.
        """
        genre_data = self.get_genre_data()
        top_genres = self.get_top_genres()

        # Generate separate graphs for each genre and save them
        for genre in top_genres:
            self.plot_genre_over_time(genre)

        # Plot all the data related to top 5 genres in one graph and save it
        plt.figure(figsize=(10, 6))
        for genre in top_genres:
            time_periods = sorted(genre_data.keys())
            genre_occurrences = [genre_data[year][genre] for year in time_periods]
            plt.plot(time_periods, genre_occurrences, marker='o', label=genre)
        plt.xlabel('Year')
        plt.ylabel('Number of Occurrences')
        plt.title('Top 5 Popular Genres Over Time')
        plt.legend()
        plt.xticks(rotation=45)
        plt.savefig('../../results/visualization/top_genres.png')
        plt.show()
        plt.close()

    def get_top_authors(self, top_n=10):
        """
        Get the top N most active authors in the dataset, excluding "Unknown" values.

        Parameters:
        top_n (int): The number of top authors to return (default is 10).

        Returns:
        pandas.Series: A Series containing the top N most active authors and their counts.
        """
        # Handle missing and "Unknown" author names
        author_data = self.data[self.data['author_name'].notna() & (self.data['author_name'] != '')]
        author_counts = author_data['author_name'].value_counts()

        # Get the top N most active authors
        top_authors = author_counts.head(top_n)

        return top_authors

    def plot_top_authors(self, top_n=10):
        """
        Plots the top N most active authors in the dataset, excluding "Unknown" values.

        Parameters:
        top_n (int): The number of top authors to plot (default is 10).
        """
        # Get the top authors
        top_authors = self.get_top_authors(top_n)

        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 6))
        top_authors.plot(kind='bar', ax=ax)

        # Set the plot title and axis labels
        ax.set_title(f'Top {top_n} Most Active Authors (Excluding "Unknown")')
        ax.set_xlabel('Author')
        ax.set_ylabel('Number of Books')

        # Rotate the x-axis labels for better readability
        plt.xticks(rotation=90)

        # Save the plot
        plt.savefig('../../results/visualization/top_authors.png')

        # Display the plot
        plt.show()

    def plot_top_ten_histogram(self):
        """
        Plots the total number of books per genre as a histogram.

        Args:
        genre_data (dict): A dictionary where keys are years and values are dictionaries
                          containing the occurrences of each genre.

        The function plots the total number of books per genre from the genre_data dictionary.
        It creates a histogram plot of the total number of books per genre for the top 10 genres.
        The plot is displayed.
        """
        genre_data = self.get_genre_data()
        # Combine genre counts across all years
        genre_counts = {}
        for year_data in genre_data.values():
            for genre, count in year_data.items():
                genre_counts[genre] = genre_counts.get(genre, 0) + count

        # Get top 10 genres by total count
        top_genres = sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)[:10]

        # Plot the histogram for top genres
        genres, counts = zip(*top_genres)
        plt.figure(figsize=(10, 6))
        plt.bar(genres, counts, color='skyblue')
        plt.xlabel('Genre')
        plt.ylabel('Total Number of Books')
        plt.title('Top 10 Genres by Total Number of Books')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('../../results/visualization/top_ten_histogram.png')
        plt.show()

    def get_genres(self):
        genres = set()
        try:
            json_dict = self.data['freebase_id_json'].iloc[0]
            for genre in json_dict.values():
                genres.add(genre)
        except (IndexError, AttributeError):
            pass
        return genres

    def get_genre_data(self):
        """
        Retrieves the occurrences of the top 5 popular genres over time.

        Returns:
            dict: A dictionary where the keys are years and the values are dictionaries
                  containing the occurrences of each of the top 5 genres.
        """
        top_genres = self.get_top_genres()
        genre_data = {}

        for index, row in self.data.iterrows():
            year = row['date']
            if year not in genre_data:
                genre_data[year] = {genre: 0 for genre in top_genres}
            for genre in top_genres:
                if isinstance(row['freebase_id_json'], str):
                    row['freebase_id_json'] = dict(row['freebase_id_json'])
                if genre in row['freebase_id_json'].values():
                    genre_data[year][genre] += 1

        return genre_data


def main():
    column_names = ["length", "freebase_id", "book_name", "author_name", "date", "freebase_id_json", "summary"]
    data = pd.read_csv('../../data/datacolab_dataset/txt_format/booksummaries.txt', sep="\t", header=None,
                       names=column_names)
    data = data.dropna(subset=['date'])
    data_cleaner = DataCleaner(data)
    data_cleaner.clean_data()
    data_utility_provider = DataUtilityProvider(data)
    data_utility_provider.unify_date_format_year()
    missing_handler = MissingHandler(data)
    missing_handler.handle_missing()
    visualizer = BookDataVisualizer(data)
    visualizer.plot_books_published_per_year()
    visualizer.plot_top_ten_histogram()
    visualizer.plot_top_genres_over_time()
    visualizer.plot_top_authors()


if __name__ == '__main__':
    main()
