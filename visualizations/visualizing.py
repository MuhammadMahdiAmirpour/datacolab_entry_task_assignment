import matplotlib.pyplot as plt
import pandas as pd

from src.data_cleaning.data_cleaning import clean_data


def plot_books_published_per_year(dataframe):
    # Count the number of books published in each year
    year_counts = dataframe['date'].value_counts().sort_index()

    # Create a bar plot
    plt.figure(figsize=(10, 6))
    plt.bar(year_counts.index, year_counts.values)
    plt.xlabel('Year')
    plt.ylabel('Number of Books Published')
    plt.title('Number of Books Published per Year')
    plt.xticks(rotation=45)
    plt.savefig('../results/visualization/books_published_per_year.png')
    plt.show()


def plot_genre_over_time(genre_data, genre):
    time_periods = sorted(genre_data.keys())
    genre_occurrences = [genre_data[year][genre] for year in time_periods]

    plt.figure(figsize=(10, 6))
    plt.plot(time_periods, genre_occurrences, marker='o')
    plt.xlabel('Year')
    plt.ylabel('Number of Occurrences')
    plt.title(f'{genre} Over Time')
    plt.xticks(rotation=45)
    filename = f'../results/visualization/{genre.replace("/", "_")}.png'  # Generate a unique filename for each genre
    plt.savefig(filename)
    plt.show()  # Show the plot
    plt.close()


def plot_top_genres_over_time(dataframe):
    # Extract genres from the 'freebase_id_json' column
    genres = []
    for json_dict in dataframe['freebase_id_json']:
        for genre in json_dict.values():
            genres.append(genre)

    # Count the number of books for each genre
    genre_counts = pd.Series(genres).value_counts()

    # Select the top 5 popular genres
    top_genres = genre_counts.nlargest(5).index.tolist()

    # Create a dictionary to store genre occurrences over time
    genre_data = {}

    # Iterate over the dataframe rows
    for index, row in dataframe.iterrows():
        year = row['date']  # Extract the year from the 'date' column
        if year not in genre_data:
            genre_data[year] = {genre: 0 for genre in top_genres}
        for genre in top_genres:
            if genre in row['freebase_id_json'].values():
                genre_data[year][genre] += 1

    # Generate separate graphs for each genre and save them
    for genre in top_genres:
        plot_genre_over_time(genre_data, genre)

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
    filename = '../results/visualization/top_genres.png'
    plt.savefig(filename)
    plt.show()  # Show the combined plot
    plt.close()


def plot_top_authors(dataframe):
    # Count the number of books published by each author
    author_counts = dataframe['author_name'].value_counts()

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
    filename = '../results/visualization/top_authors_books.png'  # Specify the filename and path
    plt.savefig(filename)  # Save the plot as an image
    plt.show()


def plot_books_per_genre(dataframe):
    # Extract genres from the 'freebase_id_json' column
    genres = []
    for json_dict in dataframe['freebase_id_json']:
        for genre in json_dict.values():
            genres.append(genre)

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

    # Save the plot as an image
    plt.savefig('../results/visualization/books_per_genre.png')

    # Display the plot
    plt.show()


def main():
    column_names = ["length", "freebase_id", "book_name", "author_name", "date", "freebase_id_json", "summary"]
    data = pd.read_csv('../data/datacolab_dataset/txt_format/booksummaries.txt', sep="\t", header=None,
                       names=column_names)
    data_cleaned = clean_data(data)

    # Call the function to plot the number of books published per year
    plot_books_published_per_year(data_cleaned)
    plot_books_per_genre(data_cleaned)
    plot_top_genres_over_time(data_cleaned)
    plot_top_authors(data_cleaned)


# Entry point of the script
if __name__ == '__main__':
    main()
