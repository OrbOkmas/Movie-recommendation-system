import os
from opustools import OpusRead


def download_opus_subtitles():
    """Systematically queries and downloads OpenSubtitles via OpusTools."""
    output_path = os.path.join("data", "raw", "opus_opensubtitles_en.txt")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Configure the OpusRead instance for English OpenSubtitles
    reader = OpusRead(
        directory="OpenSubtitles",
        source="en",
        target="en",  # Monolingual extraction
        release="v2018",  # Specify corpus version
        write=[output_path],
        write_mode="w",
        suppress_prompts=True
    )

    print("Executing OPUS extraction. This may take a while depending on corpus size...")
    reader.printPairs()
    print(f"Extraction complete. Output saved to {output_path}")


if __name__ == "__main__":
    download_opus_subtitles()