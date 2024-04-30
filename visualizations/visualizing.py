import matplotlib.pyplot as plt
import pandas as pd

from src.data_cleaning.data_cleaning import clean_data


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
        self.data = clean_data(data)

    def plot_books_published_per_year(self):
        """
        Plots the number of books published per year.

        The function counts the number of books published in each year and creates a bar plot.
        The plot is saved to the '../results/visualization/books_published_per_year.png' file and displayed.
        """
        # Count the number of books published in each year
        year_counts = self.data['date'].value_counts().sort_index()

        # Create a bar plot
        plt.figure(figsize=(10, 6))
        plt.bar(year_counts.index, year_counts.values)
        plt.xlabel('Year')
        plt.ylabel('Number of Books Published')
        plt.title('Number of Books Published per Year')
        plt.xticks(rotation=45)
        plt.savefig('../results/visualization/books_published_per_year.png')
        plt.show()

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
        filename = f'../results/visualization/{genre.replace("/", "_")}.png'
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
        plt.savefig('../results/visualization/top_genres.png')
        plt.show()
        plt.close()

    def plot_top_authors(self):
        """
        Plots the top 5 most active authors based on published books.

        The function counts the number of books published by each author, excludes the 'Unknown' author,
        and creates a bar plot of the top 5 most active authors.
        The plot is saved to the '../results/visualization/top_authors_books.png' file and displayed.
        """
        # Count the number of books published by each author
        author_counts = self.data['author_name'].value_counts()

        # Exclude unknown authors
        author_counts = author_counts[author_counts.index != 'Unknown']

        # Select the top 5 most active authors
        top_authors = author_counts.nlargest(5).index.tolist()
        book_counts = author_counts.nlargest(5).values.tolist()

        # Plot the top 5 most active authors
        plt.figure(figsize=(10, 6))
        plt.bar(top_authors, book_counts, color='skyblue')
        plt.xlabel('Authors')
        plt.ylabel('Number of Published Books')
        plt.title('Top 5 Most Active Authors Based on Published Books')
        plt.xticks(rotation=45)
        plt.savefig('../results/visualization/top_authors_books.png')
        plt.show()

    def plot_books_per_genre(self):
        """
        Plots the number of books per genre.

        The function extracts the genres from the 'freebase_id_json' column, counts the number of books for each genre,
        and creates a bar plot of the number of books per genre.
        The plot is saved to the '../results/visualization/books_per_genre.png' file and displayed.
        """
        # Extract genres from the 'freebase_id_json' column
        genres = self.get_genres()

        # Count the number of books for each genre
        genre_counts = pd.Series(genres).value_counts()

        # Filter out genres with zero occurrences
        genre_counts = genre_counts[genre_counts > 30]

        # Create a bar plot for genres with non-zero occurrences
        plt.figure(figsize=(10, 6))
        plt.bar(genre_counts.index, genre_counts.values)
        plt.xlabel('Genre')
        plt.ylabel('Number of Books')
        plt.title('Number of Books per Genre')
        plt.xticks(rotation=45)
        plt.savefig('../results/visualization/books_per_genre.png')
        plt.show()

    def get_genres(self):
        """
        Extracts the genres from the 'freebase_id_json' column.

        Returns:
            list: A list of genres extracted from the 'freebase_id_json' column.
        """
        genres = []
        for json_dict in self.data['freebase_id_json']:
            for genre in json_dict.values():
                genres.append(genre)
        return genres

    def get_top_genres(self):
        """
        Retrieves the top 5 popular genres.

        Returns:
            list: A list of the top 5 popular genres.
        """
        genres = self.get_genres()
        genre_counts = pd.Series(genres).value_counts()
        return genre_counts.nlargest(5).index.tolist()

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
                if genre in row['freebase_id_json'].values():
                    genre_data[year][genre] += 1

        return genre_data


def main():
    column_names = ["length", "freebase_id", "book_name", "author_name", "date", "freebase_id_json", "summary"]
    data = pd.read_csv('../data/datacolab_dataset/txt_format/booksummaries.txt', sep="\t", header=None,
                       names=column_names)

    visualizer = BookDataVisualizer(data)
    visualizer.plot_books_published_per_year()
    visualizer.plot_books_per_genre()
    visualizer.plot_top_genres_over_time()
    visualizer.plot_top_authors()


if __name__ == '__main__':
    main()
