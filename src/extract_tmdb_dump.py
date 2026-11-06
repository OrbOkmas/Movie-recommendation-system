import os
import requests
import pandas as pd
from datetime import datetime, timedelta


def download_tmdb_dump():
    """Downloads yesterday's TMDB movie ID dump and saves it as a CSV."""
    # Format date as MM_DD_YYYY for the TMDB URL
    yesterday = datetime.now() - timedelta(days=1)
    date_str = yesterday.strftime("%m_%d_%Y")
    url = f"http://files.tmdb.org/p/exports/movie_ids_{date_str}.json.gz"

    script_dir = os.path.dirname(os.path.abspath(__file__))
    raw_dir = os.path.join(script_dir, "..", "data", "raw")
    os.makedirs(raw_dir, exist_ok=True)

    gz_path = os.path.join(raw_dir, f"movie_ids_{date_str}.json.gz")
    csv_path = os.path.join(raw_dir, "tmdb_universe.csv")

    print(f"Downloading TMDB export from {url}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()

    with open(gz_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    print("Parsing JSON Lines into a dataframe...")
    # The file is in JSON Lines format (one JSON object per line)
    df = pd.read_json(gz_path, lines=True, compression='gzip')

    # Exporting with specific encoding and delimiter for smooth cross-platform handoffs
    df.to_csv(csv_path, index=False, sep=';', encoding='utf-8-sig')
    print(f"Extraction complete. {len(df)} records saved to {csv_path}")

    # Clean up the raw .gz file to save disk space
    os.remove(gz_path)


if __name__ == "__main__":
    download_tmdb_dump()