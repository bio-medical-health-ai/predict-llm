"""Utility functions for preparing data for predict_llm."""

from __future__ import annotations

import importlib.util
import json
import os
import re
import subprocess

from chonkie import SDPMChunker
from tqdm import tqdm


def initialize():
    """Ensure required models are downloaded before using the package."""
    # Check if huggingface_hub is installed
    if importlib.util.find_spec('huggingface_hub') is None:
        subprocess.run(['pip', 'install', 'huggingface_hub'], check=True)

    # Download download_models_hf.py script if it does not exist
    script_path = os.path.join(
        os.path.dirname(__file__),
        'download_models_hf.py',
    )
    if not os.path.exists(script_path):
        subprocess.run(
            [
                'wget',
                'https://github.com/opendatalab/MinerU/raw/master/scripts/download_models_hf.py',
                '-O',
                script_path,
            ],
            check=True,
        )

    subprocess.run(['python', script_path], check=True)

    print('Initialization complete. Models are ready for use.')
    print(
        "Configuration file 'magic-pdf.json' is located in the user directory."
        ' Modify it to enable or disable features like CUDA acceleration.',
    )


def pdf_parser(input_pdf: str, output_path: str):
    """Run the magic-pdf command to process the input PDF.

    Output the result to the specified path.

    :param input_pdf: Path to the input PDF file. It can be folder or file.
    :param output_path: Path to the output file or directory.
    """
    command = ['magic-pdf', '-p', input_pdf, '-o', output_path]

    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
        print(f'Command executed successfully: {command}')
        print('Processing the input PDF(s)')
    except subprocess.CalledProcessError as e:
        print(f'Error executing command: {e.stderr}')


def combine_texts_in_directory(input_dir, output_dir):
    """Traverse directory to find and combine content from JSON files.

    Find all `_content_list.json` files, combine `text` fields from them,
    and save the output.

    Args:
        input_dir (str): Input directory with `_content_list.json` files.
        output_dir (str): Directory to save combined JSON files.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith('_content_list.json'):
                input_file_path = os.path.join(root, file)
                output_file_name = (
                    f'{file.replace("_content_list.json", "")}.txt'
                )
                output_file_path = os.path.join(output_dir, output_file_name)

                try:
                    with open(input_file_path, encoding='utf-8') as f:
                        data = json.load(f)

                    # Combine all "text" fields where "type" is "text"
                    combined_texts = '\n'.join(
                        item['text']
                        for item in data
                        if item.get('type') == 'text'
                    )

                    # Write the combined text to the output file
                    with open(
                        output_file_path,
                        'w',
                        encoding='utf-8',
                    ) as out_f:
                        out_f.write(combined_texts)

                    print(
                        f'Processed: {input_file_path} -> {output_file_path}',
                    )
                except Exception as e:
                    print(f'Error processing {input_file_path}: {e}')


def sanitize_filename(filename):
    """Sanitize filename by removing special characters and formatting.

    1. Removing special characters
    2. Replacing spaces with underscores
    3. Handling parentheses
    """
    # Remove or replace special characters
    filename = re.sub(r'[^\w\s-]', '_', filename)
    # Replace spaces with underscores
    filename = re.sub(r'\s+', '_', filename)
    # Remove multiple consecutive underscores
    filename = re.sub(r'_+', '_', filename)
    # Remove leading/trailing underscores
    filename = filename.strip('_')
    return filename


def initialize_chunker():
    """Initialize SDPMChunker with default configuration.

    Returns:
        SDPMChunker: Configured chunker instance.
    """
    return SDPMChunker(
        embedding_model='minishlab/potion-base-8M',
        threshold=0.5,
        chunk_size=512,
        min_sentences=1,
        skip_window=1,
    )


def _process_batch(batch_files, chunker, output_dir):
    """Process a batch of files and save chunks.

    Args:
        batch_files: List of files to process
        chunker: SDPMChunker instance
        output_dir: Output directory for chunks
    """
    batch_texts = []
    file_names = []

    for file_path in batch_files:
        with open(file_path, encoding='utf-8') as f:
            text = f.read()
            batch_texts.append(text)
            original_name = os.path.basename(file_path)
            sanitized_name = sanitize_filename(original_name)
            file_names.append(sanitized_name)

    batch_chunks = chunker.chunk_batch(batch_texts)

    for sanitized_name, chunks in zip(file_names, batch_chunks):
        records = []
        for chunk_idx, chunk in enumerate(chunks):
            if not chunk.text.strip():
                continue

            record = {
                'id': f'{sanitized_name}_{chunk_idx}',
                'contents': chunk.text,
                'token_count': chunk.token_count,
                'sentence_count': len(chunk.sentences),
            }
            records.append(record)

        if records:
            output_file = os.path.join(
                output_dir,
                f'{sanitized_name.replace(".txt", "")}.jsonl',
            )
            with open(output_file, 'w', encoding='utf-8') as out_f:
                for record in records:
                    out_f.write(json.dumps(record, ensure_ascii=False) + '\n')


def process_files(input_dir, output_dir, chunker=None, batch_size=10):
    """Process text files in batches and generate chunked output.

    Args:
        input_dir: Input directory containing text files
        output_dir: Output directory for processed chunks
        chunker: Optional SDPMChunker instance
        batch_size: Number of files to process per batch
    """
    if chunker is None:
        chunker = initialize_chunker()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    txt_files = []
    for root, _, files in os.walk(input_dir):
        txt_files.extend(
            os.path.join(root, f) for f in files if f.endswith('.txt')
        )

    for i in tqdm(
        range(0, len(txt_files), batch_size),
        desc='Processing batches',
        unit='batch',
    ):
        batch_files = txt_files[i : i + batch_size]
        _process_batch(batch_files, chunker, output_dir)


def synthetic_data_generator() -> None:
    """Generate synthetic data."""
    print('Generating synthetic data...')
