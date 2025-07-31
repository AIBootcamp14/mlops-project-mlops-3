import pandas as pd
from dotenv import load_dotenv

from preprocessing import TMDBDataPreprocessor
from crawler import TMDBCrawler

load_dotenv()


def run_popular_movie_crawler():
    tmdb_crawler = TMDBCrawler()
    result = tmdb_crawler.get_bulk_popular_movies(start_page=1, end_page=50)
    tmdb_crawler.save_movies_to_json_file(result, "./result", "popular")

    tmdb_preprocessor = TMDBDataPreprocessor()
    preprocessed_result = tmdb_preprocessor.run_full_pipeline("./result/popular.json")
    if preprocessed_result:
        tmdb_preprocessor.save_processed_data(preprocessed_result)


if __name__ == '__main__':
    run_popular_movie_crawler()