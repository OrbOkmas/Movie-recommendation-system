import os
import time
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
TMDB_TOKEN = os.getenv("TMDB_BEARER_TOKEN")


def enrich_tmdb_details():
    if not TMDB_TOKEN:
        raise ValueError("TMDB_BEARER_TOKEN is missing from your .env file.")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    processed_dir = os.path.join(script_dir, "..", "data", "processed")

    aligned_path = os.path.join(processed_dir, "aligned_metadata.csv")
    cache_path = os.path.join(processed_dir, "tmdb_details_cache.csv")
    checked_log_path = os.path.join(processed_dir, "checked_tmdb_details.txt")
    final_output_path = os.path.join(processed_dir, "enriched_metadata.csv")

    print("Loading aligned metadata...")
    aligned_df = pd.read_csv(aligned_path, sep=';', encoding='utf-8-sig')

    # Ensure tmdb_id is strictly an integer (prevents float formatting like '550.0' causing 404s)
    aligned_df['tmdb_id'] = pd.to_numeric(aligned_df['tmdb_id'], errors='coerce')
    aligned_df = aligned_df.dropna(subset=['tmdb_id']).copy()
    aligned_df['tmdb_id'] = aligned_df['tmdb_id'].astype(int)

    unique_tmdb_ids = aligned_df['tmdb_id'].drop_duplicates().tolist()
    print(f"Total unique matched TMDB movies: {len(unique_tmdb_ids)}")

    # Load checkpoint state (both cached details and 404/checked IDs)
    checked_ids = set()
    if os.path.exists(cache_path):
        cached_df = pd.read_csv(cache_path, sep=';', encoding='utf-8-sig')
        checked_ids.update(pd.to_numeric(cached_df['tmdb_id'], errors='coerce').dropna().astype(int).tolist())
    if os.path.exists(checked_log_path):
        with open(checked_log_path, 'r', encoding='utf-8') as f:
            for line in f:
                val = line.strip()
                if val.isdigit():
                    checked_ids.add(int(val))

    to_process = [tid for tid in unique_tmdb_ids if tid not in checked_ids]
    print(f"Previously checked TMDB IDs: {len(checked_ids)}")
    print(f"Remaining TMDB IDs to enrich: {len(to_process)}")

    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {TMDB_TOKEN}"
    }

    batch_records = []
    checked_batch = []

    # Use a persistent HTTP Session for connection pooling across thousands of calls
    with requests.Session() as session:
        session.headers.update(headers)

        for idx, tmdb_id in enumerate(to_process, start=1):
            url = f"https://api.themoviedb.org/3/movie/{tmdb_id}?append_to_response=keywords"

            try:
                response = session.get(url, timeout=10)

                # Handle potential rate-limit burst with a brief pause and single retry
                if response.status_code == 429:
                    time.sleep(2.0)
                    response = session.get(url, timeout=10)

                if response.status_code == 200:
                    data = response.json()

                    # Format genres and keywords as pipe-delimited strings to avoid CSV delimiter conflicts
                    genres_list = [g.get('name', '').strip() for g in data.get('genres', []) if g.get('name')]
                    keywords_raw = data.get('keywords', {}).get('keywords', [])
                    keywords_list = [k.get('name', '').strip() for k in keywords_raw if k.get('name')]

                    batch_records.append({
                        'tmdb_id': tmdb_id,
                        'genres': "|".join(genres_list),
                        'keywords': "|".join(keywords_list),
                        'original_language': data.get('original_language', ''),
                        'runtime': data.get('runtime'),
                        'vote_average': data.get('vote_average'),
                        'vote_count': data.get('vote_count')
                    })
                    checked_batch.append(str(tmdb_id))

                elif response.status_code == 404:
                    # Log missing/deleted TMDB IDs so restarts do not re-query them
                    checked_batch.append(str(tmdb_id))

            except requests.exceptions.RequestException:
                print(f"Network error on tmdb_id={tmdb_id}. Will retry on next run.")

            # Flush checkpoints every 100 requests
            if idx % 100 == 0:
                _flush_checkpoints(cache_path, checked_log_path, batch_records, checked_batch)
                batch_records = []
                checked_batch = []
                print(f"Enriched {idx} / {len(to_process)} remaining movies...")
                time.sleep(0.05)

    # Flush remaining items (< 100)
    _flush_checkpoints(cache_path, checked_log_path, batch_records, checked_batch)

    # Merge cached details back onto aligned_metadata.csv
    if os.path.exists(cache_path):
        print("Merging enriched details with aligned_metadata.csv...")
        final_cache_df = pd.read_csv(cache_path, sep=';', encoding='utf-8-sig')
        final_cache_df['tmdb_id'] = pd.to_numeric(final_cache_df['tmdb_id'], errors='coerce')
        final_cache_df = final_cache_df.dropna(subset=['tmdb_id']).drop_duplicates(subset=['tmdb_id'], keep='first')
        final_cache_df['tmdb_id'] = final_cache_df['tmdb_id'].astype(int)

        enriched_df = pd.merge(aligned_df, final_cache_df, on='tmdb_id', how='left')
        enriched_df.to_csv(final_output_path, index=False, sep=';', encoding='utf-8-sig')
        print(f"Complete! Saved {len(enriched_df)} enriched movies to {final_output_path}")


def _flush_checkpoints(cache_path, checked_log_path, batch_records, checked_batch):
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    if batch_records:
        batch_df = pd.DataFrame(batch_records)
        write_header = not os.path.exists(cache_path)
        batch_df.to_csv(cache_path, mode='a', index=False, sep=';', encoding='utf-8-sig', header=write_header)

    if checked_batch:
        with open(checked_log_path, 'a', encoding='utf-8') as f:
            f.write("\n".join(checked_batch) + "\n")


if __name__ == "__main__":
    enrich_tmdb_details()