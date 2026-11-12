import os
import zipfile
import xml.etree.ElementTree as et
import pandas as pd


def parse_opus_zip():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    zip_path = os.path.join(script_dir, "..", "data", "raw", "en.zip")
    csv_output_path = os.path.join(script_dir, "..", "data", "raw", "opus_metadata.csv")
    text_output_path = os.path.join(script_dir, "..", "data", "raw", "raw_subtitles.txt")

    metadata_records = []

    print(f"Opening {zip_path} directly...")
    with zipfile.ZipFile(zip_path, 'r') as z:
        xml_files = [f for f in z.namelist() if f.endswith('.xml')]
        print(f"Found {len(xml_files)} subtitle files. Processing metadata from folder paths...")

        with open(text_output_path, 'w', encoding='utf-8') as text_out:
            for idx, file_name in enumerate(xml_files):
                # Extract metadata from OPUS path structure: lang / year / imdb_id / file.xml
                parts = file_name.split('/')

                if len(parts) >= 4:
                    year = parts[-3]
                    imdb_raw = parts[-2]
                    # Standard IMDB IDs are padded to at least 7 digits (e.g., 749648 -> 0749648)
                    imdb_id = f"tt{imdb_raw.zfill(7)}"
                else:
                    year = ""
                    imdb_id = ""

                opus_id = f"opus_{idx}"
                metadata_records.append({
                    'opus_id': opus_id,
                    'file_path': file_name,
                    'title': f"Unknown_{opus_id}",  # To be replaced by Week 2 TMDB alignment
                    'release_year': year,
                    'imdb_id': imdb_id
                })

                # Extract subtitle text
                try:
                    with z.open(file_name) as f:
                        tree = et.parse(f)
                        root = tree.getroot()

                        # Your diagnostic snippet showed dialogue is inside <w> tags
                        sentences = [str(w.text) for w in root.findall('.//s/w') if w.text]
                        if sentences:
                            full_text = " ".join(sentences)
                            text_out.write(f"--- MOVIE_ID: {opus_id} ---\n{full_text}\n\n")

                except (et.ParseError, zipfile.BadZipFile, AttributeError, KeyError):
                    continue

                if (idx + 1) % 5000 == 0:
                    print(f"Processed {idx + 1} / {len(xml_files)} files...")

    df = pd.DataFrame(metadata_records)
    os.makedirs(os.path.dirname(csv_output_path), exist_ok=True)
    df.to_csv(csv_output_path, index=False, sep=';', encoding='utf-8-sig')
    print(f"Parsing complete! Saved metadata to {csv_output_path}")


if __name__ == "__main__":
    parse_opus_zip()