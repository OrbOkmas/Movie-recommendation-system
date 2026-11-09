import os
import glob
import xml.etree.ElementTree as ET
import pandas as pd


def extract_metadata_from_xml():
    """Parses OPUS OpenSubtitles XML headers for Title, Year, and IMDB ID."""

    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Point this to where opustools downloaded the raw XML files
    # Default is usually within your working directory under OpenSubtitles/raw/en/
    xml_directory = os.path.join(script_dir, "..", "OpenSubtitles", "raw", "en")
    output_path = os.path.join(script_dir, "..", "data", "raw", "opus_metadata.csv")

    # Use glob to recursively find all .xml files in the directory
    xml_files = glob.glob(os.path.join(xml_directory, "**", "*.xml"), recursive=True)

    print(f"Found {len(xml_files)} XML files. Parsing metadata...")

    metadata_records = []

    for file_path in xml_files:
        try:
            # Parse the XML file
            tree = ET.parse(file_path)
            root = tree.getroot()

            # OpenSubtitles XMLs usually store metadata in the <meta> block
            # Note: Exact tag paths can vary slightly by OPUS corpus version
            meta = root.find('meta')
            if meta is not None:
                title_node = meta.find('source_file')  # Often contains the original file name/title
                year_node = meta.find('year')
                imdb_node = meta.find('imdb_id')

                # Extract text if the node exists, else leave blank
                title = title_node.text if title_node is not None else "Unknown"
                year = year_node.text if year_node is not None else ""
                imdb_id = imdb_node.text if imdb_node is not None else ""

                # Clean up the imdb_id (sometimes they include 'tt', sometimes they don't)
                if imdb_id and not imdb_id.startswith('tt'):
                    imdb_id = f"tt{imdb_id}"

                metadata_records.append({
                    "opus_id": os.path.basename(file_path).replace('.xml', ''),
                    "title": title,
                    "release_year": year,
                    "imdb_id": imdb_id
                })

        except ET.ParseError:
            # Skip corrupted XML files without crashing the pipeline
            continue

    # Load into Pandas and export for the team
    df = pd.DataFrame(metadata_records)

    # Clean up the title strings (removing file extensions if present)
    df['title'] = df['title'].astype(str).str.replace(r'\.srt$|\.sub$', '', regex=True)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False, sep=';', encoding='utf-8-sig')

    print(f"Metadata extraction complete. Saved {len(df)} records to {output_path}")


if __name__ == "__main__":
    extract_metadata_from_xml()