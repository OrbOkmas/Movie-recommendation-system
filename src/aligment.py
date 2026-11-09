import os
import time
import pandas as pd
import tmdbsimple as tmdb
from dotenv import load_dotenv

# Automatically loads the TMDB API key from your PyCharm .env configuration
load_dotenv()
tmdb.API_KEY = os.getenv("TMDB_API_KEY")


def align_via_imdb():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    opus_path = os.path.join(script_dir, "..", "data", "raw", "opus_metadata.csv")
    output_path = os.path.join(script_dir, "..", "data", "processed", "aligned_metadata.csv")

    print("Loading OPUS metadata...")
    df = pd.read_csv(opus_path, sep=';')

    # Filter for rows that actually contain a valid IMDB ID (starts with 'tt')
    valid_df = df[df['imdb_id'].notna() & df['imdb_id'].str.startswith('tt')].copy()
    print(f"Found {len(valid_df)} records with valid IMDB IDs ready for alignment.")

    aligned_records = []

    print("Querying TMDB /find endpoint. This will take time...")

    # Track iterations for progress reporting
    count = 0

    for _, row in valid_df.iterrows():
        imdb_id = row['imdb_id']
        try:
            # Ping the TMDB API for this specific IMDB ID
            find = tmdb.Find(imdb_id)
            response = find.info(external_source='imdb_id')

            # The API returns a dictionary of lists. We want 'movie_results'.
            if response['movie_results']:
                movie = response['movie_results'][0]  # Take the exact match

                aligned_records.append({
                    'opus_id': row['opus_id'],
                    'imdb_id': imdb_id,
                    'tmdb_id': movie['id'],
                    'tmdb_title': movie['title'],
                    'release_year': movie.get('release_date', '')[:4],
                    'plot': movie.get('overview', '')
                })
        except Exception as e:
            # Silently catch timeouts or API errors so the script doesn't crash halfway
            pass

        count += 1
        # Print progress and throttle the requests so TMDB doesn't ban your IP
        if count % 100 == 0:
            print(f"Processed {count} movies...")
            time.sleep(0.2)  # 200ms delay to respect API rate limits

    aligned_df = pd.DataFrame(aligned_records)

    # Export for team handoff
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    aligned_df.to_csv(output_path, index=False, sep=';', encoding='utf-8-sig')
    print(f"Alignment complete. {len(aligned_df)} records perfectly matched and saved to {output_path}")


if __name__ == "__main__":
    # TEST MODE: Uncomment the line below to test on just the first 10 rows before running the full dataset
    # valid_df = valid_df.head(10)
    align_via_imdb()