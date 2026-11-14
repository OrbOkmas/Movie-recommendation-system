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

    checked_log_path = os.path.join(script_dir, "..", "data", "processed", "checked_imdb_ids.txt")

    print("Loading OPUS metadata...")
    df = pd.read_csv(opus_path, sep=';')
    valid_df = df[df['imdb_id'].notna() & df['imdb_id'].astype(str).str.startswith('tt')].copy()
    valid_df = valid_df.drop_duplicates(subset=['imdb_id'], keep='first').copy()
    print(f"Unique movies in OPUS metadata: {len(valid_df)}")

    # Load all previously checked IDs (both CSV matches and non-movie misses)
    processed_ids = set()
    if os.path.exists(output_path):
        existing_df = pd.read_csv(output_path, sep=';', encoding='utf-8-sig')
        processed_ids.update(existing_df['imdb_id'].dropna().astype(str).tolist())
    if os.path.exists(checked_log_path):
        with open(checked_log_path, 'r', encoding='utf-8') as f:
            processed_ids.update(line.strip() for line in f if line.strip())

    print(f"Found {len(processed_ids)} previously checked IDs. Resuming...")
    to_process = valid_df[~valid_df['imdb_id'].isin(processed_ids)]
    print(f"Movies left to process: {len(to_process)}")

    # --- TIP 1: v4 BEARER TOKEN HEADERS ---
    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {TMDB_TOKEN}"
    }

    aligned_records = []
    checked_batch = []
    count = 0

    for _, row in to_process.iterrows():
        imdb_id = row['imdb_id']
        url = f"https://api.themoviedb.org/3/find/{imdb_id}?external_source=imdb_id"

        try:
            response = requests.get(url, headers=headers, timeout=10)

            if response.status_code == 200:
                # Record that this ID was successfully queried (even if it's a TV show with no movie_results)
                checked_batch.append(imdb_id)
                data = response.json()

                if data.get('movie_results'):
                    movie = data['movie_results'][0]
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

        count += 1

        # Batch save every 100 API calls
        if count % 100 == 0:
            if aligned_records:
                batch_df = pd.DataFrame(aligned_records)
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                write_header = not os.path.exists(output_path)
                batch_df.to_csv(output_path, mode='a', index=False, sep=';', encoding='utf-8-sig', header=write_header)
                aligned_records = []

            if checked_batch:
                os.makedirs(os.path.dirname(checked_log_path), exist_ok=True)
                with open(checked_log_path, 'a', encoding='utf-8') as f:
                    f.write("\n".join(checked_batch) + "\n")
                checked_batch = []

            print(f"Checked {count} / {len(to_process)} remaining IDs...")
            time.sleep(0.05)

    # Final flush for the last partial batch (< 100 items)
    if aligned_records:
        batch_df = pd.DataFrame(aligned_records)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        write_header = not os.path.exists(output_path)
        batch_df.to_csv(output_path, mode='a', index=False, sep=';', encoding='utf-8-sig', header=write_header)

    if checked_batch:
        os.makedirs(os.path.dirname(checked_log_path), exist_ok=True)
        with open(checked_log_path, 'a', encoding='utf-8') as f:
            f.write("\n".join(checked_batch) + "\n")

    print(f"Alignment complete. Master lookup table is at {output_path}")


if __name__ == "__main__":
    align_via_imdb()