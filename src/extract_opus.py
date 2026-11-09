import os
import requests


def download_opus_xml_zip():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    raw_dir = os.path.join(script_dir, "..", "data", "raw")
    os.makedirs(raw_dir, exist_ok=True)

    zip_path = os.path.join(raw_dir, "en.zip")
    url = "https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2018/xml/en.zip"

    print(f"Bypassing opustools. Downloading 24GB OPUS English XML directly from {url}...")

    # stream=True is critical to prevent the 24GB file from crashing your RAM
    with requests.get(url, stream=True) as response:
        response.raise_for_status()

        with open(zip_path, 'wb') as f:
            # Writing in 8MB chunks
            for chunk in response.iter_content(chunk_size=8192 * 1024):
                if chunk:
                    f.write(chunk)

    print(f"Download complete. Saved to {zip_path}")


if __name__ == "__main__":
    download_opus_xml_zip()