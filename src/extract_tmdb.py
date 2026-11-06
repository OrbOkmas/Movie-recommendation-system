import os
import json
import tmdbsimple as tmdb
from dotenv import load_dotenv

# When running in PyCharm with the plugin, os.environ already has the key.
# load_dotenv() acts as a safety net for terminal/command-line execution.
load_dotenv()
# os.getenv will safely pull the key regardless of which method injected it
tmdb.API_KEY = os.getenv("TMDB_API_KEY")


def fetch_movie_metadata(start_page=1, end_page=5):
    """Scrapes TMDB for ID, Title, Release Year, and Plot."""
    movies = tmdb.Movies()
    extracted_data = []

    for page in range(start_page, end_page + 1):
        response = movies.popular(page=page)

        for movie in response['results']:
            # Extract only the required fields
            extracted_data.append({
                "id": movie.get('id'),
                "title": movie.get('title'),
                "release_year": movie.get('release_date', '')[:4],
                "plot": movie.get('overview', '')
            })

    return extracted_data


if __name__ == "__main__":
    data = fetch_movie_metadata(start_page=1, end_page=1)

    # Save locally to data/raw/
    output_path = os.path.join("data", "raw", "tmdb_metadata.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
    print(f"Successfully extracted {len(data)} TMDB records.")