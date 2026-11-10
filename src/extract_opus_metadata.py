import os
import zipfile
import xml.etree.ElementTree as et
import pandas as pd


def parse_opus_zip():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    zip_path = os.path.join(script_dir, "..", "data", "raw", "en.zip")
    csv_output_path = os.path.join(script_dir, "..", "data", "raw", "opus_metadata.csv")
    text_output_path = os.path.join(script_dir, "..", "data", "raw", "raw_subtitles.txt")

    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"Could not find {zip_path}. Run extract_opus.py first.")

    metadata_records = []

    print(f"Opening {zip_path} directly...")
    with zipfile.ZipFile(zip_path, 'r') as z:
        xml_files = [f for f in z.namelist() if f.endswith('.xml')]
        print(f"Found {len(xml_files)} subtitle files in zip archive. Processing...")

        with open(text_output_path, 'w', encoding='utf-8') as text_out:
            for idx, file_name in enumerate(xml_files):
                try:
                    with z.open(file_name) as f:
                        tree = et.parse(f)
                        root = tree.getroot()

                        # Extract Metadata from XML Header
                        meta = root.find('meta')
                        imdb_id = ""
                        title = os.path.basename(file_name).replace('.xml', '')
                        year = ""

                        if meta is not None:
                            imdb_node = meta.find('.//imdb_id')
                            if imdb_node is not None and imdb_node.text:
                                raw_imdb = imdb_node.text.strip()
                                imdb_id = raw_imdb if raw_imdb.startswith('tt') else f"tt{raw_imdb}"

                            year_node = meta.find('.//year')
                            if year_node is not None and year_node.text:
                                year = year_node.text.strip()

                        opus_id = f"opus_{idx}"
                        metadata_records.append({
                            'opus_id': opus_id,
                            'file_path': file_name,
                            'title': title,
                            'release_year': year,
                            'imdb_id': imdb_id
                        })

                        # Explicitly cast s.text to str to satisfy PyCharm type checking
                        sentences = [str(s.text) for s in root.findall('.//s/w') if s.text]
                        if sentences:
                            full_text = " ".join(sentences)
                            text_out.write(f"--- MOVIE_ID: {opus_id} ---\n{full_text}\n\n")

                # Catch specific parsing errors rather than broad Exception
                except (et.ParseError, zipfile.BadZipFile, AttributeError, KeyError):
                    continue

                if (idx + 1) % 5000 == 0:
                    print(f"Processed {idx + 1} / {len(xml_files)} files...")

    # Export metadata CSV formatted for team handoffs
    df = pd.DataFrame(metadata_records)
    os.makedirs(os.path.dirname(csv_output_path), exist_ok=True)
    df.to_csv(csv_output_path, index=False, sep=';', encoding='utf-8-sig')
    print(f"Parsing complete! Saved metadata to {csv_output_path} and dialogue text to {text_output_path}")


if __name__ == "__main__":
    parse_opus_zip()