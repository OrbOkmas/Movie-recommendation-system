import os
import time
import requests
import pandas as pd
from dotenv import load_dotenv

# Load the v4 Bearer Token from .env
load_dotenv()
TMDB_TOKEN = os.getenv("TMDB_BEARER_TOKEN")


def align_via_imdb():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    opus_path = os.path.join(script_dir, "..", "data", "raw", "opus_metadata.csv")
    output_path = os.path.join(script_dir, "..", "data", "processed", "aligned_metadata.csv")

    print("Loading OPUS metadata...")
    df = pd.read_csv(opus_path, sep=';')
    valid_df = df[df['imdb_id'].notna() & df['imdb_id'].astype(str).str.startswith('tt')].copy()
    print(f"Total raw rows with valid IMDB IDs: {len(valid_df)}")

    # Deduplicate directly into valid_df so every downstream step uses unique movies
    valid_df = valid_df.drop_duplicates(subset=['imdb_id'], keep='first').copy()
    print(f"Unique movies to query on TMDB: {len(valid_df)}")

    # Checkpointing: filter against the deduplicated valid_df
    if os.path.exists(output_path):
        existing_df = pd.read_csv(output_path, sep=';', encoding='utf-8-sig')
        processed_ids = set(existing_df['imdb_id'].astype(str).tolist())
        print(f"Found {len(processed_ids)} previously processed movies. Resuming...")
    else:
        processed_ids = set()
        print("Starting fresh alignment...")

    to_process = valid_df[~valid_df['imdb_id'].isin(processed_ids)]
    print(f"Movies left to process: {len(to_process)}")

    # --- TIP 1: v4 BEARER TOKEN HEADERS ---
    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {TMDB_TOKEN}"
    }

    aligned_records = []
    count = 0

    for _, row in to_process.iterrows():
        imdb_id = row['imdb_id']
        url = f"https://api.themoviedb.org/3/find/{imdb_id}?external_source=imdb_id"

        try:
            response = requests.get(url, headers=headers, timeout=10)

            if response.status_code == 200:
                data = response.json()
                if data.get('movie_results'):
                    movie = data['movie_results'][0]

                    # --- TIP 3: HANDLING IMAGES PROPERLY ---
                    poster_path = movie.get('poster_path')
                    full_poster_url = f"https://image.tmdb.org/t/p/w500{poster_path}" if poster_path else ""

                    aligned_records.append({
                        'opus_id': row['opus_id'],
                        'imdb_id': imdb_id,
                        'tmdb_id': movie.get('id'),
                        'tmdb_title': movie.get('title'),
                        'release_year': str(movie.get('release_date', ''))[:4],
                        'poster_url': full_poster_url,
                        'plot': movie.get('overview', '')
                    })
        except requests.exceptions.RequestException:
            print(f"Network error on {imdb_id}. Skipping for now.")
            pass  # Catch timeouts without crashing

        count += 1

        # Batch save every 100 records to protect progress
        if count % 100 == 0:
            batch_df = pd.DataFrame(aligned_records)
            # If file doesn't exist yet, write headers. If it does, append without headers.
            write_header = not os.path.exists(output_path)
            batch_df.to_csv(output_path, mode='a', index=False, sep=';', encoding='utf-8-sig', header=write_header)

            aligned_records = []  # Clear the batch from memory
            print(f"Processed and saved {count} movies...")

            # Tiny sleep just to be polite to TMDB's servers
            time.sleep(0.05)

    # Save any remaining records in the final batch
    if aligned_records:
        batch_df = pd.DataFrame(aligned_records)
        write_header = not os.path.exists(output_path)
        batch_df.to_csv(output_path, mode='a', index=False, sep=';', encoding='utf-8-sig', header=write_header)

    print(f"Alignment complete. Master lookup table is at {output_path}")


if __name__ == "__main__":
    align_via_imdb()