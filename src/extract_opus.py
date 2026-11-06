import os
from opustools import OpusRead


def download_opus_subtitles():
    """Systematically queries and downloads OpenSubtitles via OpusTools."""
    # Anchor the path to the project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, "..", "data", "raw", "opus_opensubtitles_en.txt")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Configure the OpusRead instance for English OpenSubtitles
    reader = OpusRead(
        directory="OpenSubtitles",
        source="en",
        target="en",  # Monolingual extraction
        release="v2018",  # Specify corpus version
        write=[output_path],
        write_mode="moses",
        maximum=1000,  # Stops after processing 1000 alignments
        suppress_prompts=True
    )
    '''
    The 'moses' format is exactly what you need for your workflow. 
    It is the standard format for Natural Language Processing because it extracts pure, 
    plain text (one sentence per line) and strips out the heavy XML structure, 
    preparing it perfectly for your spaCy cleaning pipeline
    '''
    print("Executing OPUS extraction. This may take a while depending on corpus size...")
    reader.printPairs()
    print(f"Extraction complete. Output saved to {output_path}")


if __name__ == "__main__":
    download_opus_subtitles()