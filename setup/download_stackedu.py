"""
Example script to download code content.

Usage:
    python example_download.py --num_samples 1000 --output data/sample.jsonl
"""

import argparse
import gzip
import json
from pathlib import Path
import os
import boto3
from botocore.exceptions import ClientError
from datasets import load_dataset, concatenate_datasets
from botocore import UNSIGNED
from botocore.config import Config
# Constants
BUCKET_NAME = "softwareheritage"
DATASET_NAME = "HuggingFaceTB/stack-edu"
CACHE_DIR = os.environ.get("HF_HOME", ".hf_cache")
NUM_PROC = 16

# Language configs with their num_examples (from dataset_info)
# Used to maintain proportional sampling across languages
LANGUAGE_NUM_EXAMPLES = {
    "C": 5_848_375,
    "CSharp": 11_425_016,
    "Cpp": 16_246_746,
    "Go": 1_917_163,
    "Java": 44_990_158,
    "JavaScript": 13_253_431,
    "Markdown": 20_687_077,
    "PHP": 9_914_497,
    "Python": 25_286_019,
    "Ruby": 2_976_874,
    "Rust": 1_135_379,
    "SQL": 2_504_412,
    "Shell": 4_133_547,
    "Swift": 2_454_309,
    "TypeScript": 4_290_356,
}

# Initialize S3 client
# s3 = boto3.client("s3")
s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))


def download_contents(blob_id: str) -> dict:
    """Download and decompress content from S3 given a blob_id."""
    key = f"content/{blob_id}"
    try:
        obj = s3.get_object(Bucket=BUCKET_NAME, Key=key)
        with gzip.GzipFile(fileobj=obj["Body"]) as fin:
            content = fin.read().decode("utf-8", errors="ignore")
        return {"text": content, "download_success": True}
    except ClientError as e:
        if e.response["Error"]["Code"] == "NoSuchKey":
            print(f"File not found: {key}")
            return {"text": "", "download_success": False}
        else:
            raise


def save_to_jsonl(dataset, output_path: str) -> None:
    """Save dataset to a JSONL file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for example in dataset:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")

    print(f"Saved {len(dataset)} examples to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare Stack-edu data as an example from Software Heritage S3 bucket."
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1000,
        help="Total number of samples to download (default: 1000)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path for the JSONL file",
    )
    parser.add_argument(
        "--languages",
        type=str,
        nargs="+",
        default=None,
        help="Specific languages to sample from (default: all languages)",
    )
    args = parser.parse_args()

    # Determine which languages to sample
    if args.languages:
        languages = args.languages
        # Validate language names
        for lang in languages:
            if lang not in LANGUAGE_NUM_EXAMPLES:
                raise ValueError(
                    f"Unknown language: {lang}. "
                    f"Available: {list(LANGUAGE_NUM_EXAMPLES.keys())}"
                )
    else:
        languages = list(LANGUAGE_NUM_EXAMPLES.keys())

    # Compute samples per language (maintaining ratios)
    # Recompute ratios based on selected languages only
    selected_totals = {lang: LANGUAGE_NUM_EXAMPLES[lang] for lang in languages}
    total_selected = sum(selected_totals.values())
    
    raw_samples = {
        lang: (count / total_selected) * args.num_samples
        for lang, count in selected_totals.items()
    }
    samples_per_lang = {lang: int(s) for lang, s in raw_samples.items()}
    
    # Distribute remaining samples by largest remainder
    remainder = args.num_samples - sum(samples_per_lang.values())
    remainders = [(lang, raw_samples[lang] - samples_per_lang[lang]) for lang in samples_per_lang]
    remainders.sort(key=lambda x: x[1], reverse=True)
    for i in range(remainder):
        samples_per_lang[remainders[i][0]] += 1

    # Print sampling plan
    print(f"Sampling plan for {args.num_samples} total samples:")
    print("-" * 40)
    for lang, count in sorted(samples_per_lang.items(), key=lambda x: -x[1]):
        pct = (count / args.num_samples) * 100
        print(f"  {lang:12s}: {count:6d} samples ({pct:5.2f}%)")
    print("-" * 40)

    # Load and sample from each language config
    all_datasets = []
    for lang, num_samples in samples_per_lang.items():
        if num_samples == 0:
            continue
        print(f"\nDownloading {lang}...")
        ds = load_dataset(
            DATASET_NAME,
            name=lang,  # config_name for the language
            split="train",
            num_proc=NUM_PROC,
            cache_dir=CACHE_DIR,
        )
        # Shuffle and take samples
        ds = ds.shuffle(seed=42).select(range(min(num_samples, len(ds))))
        all_datasets.append(ds)
        print(f"  Loaded {len(ds)} samples from {lang}")

    # Concatenate all language datasets
    print("\nConcatenating datasets...")
    ds = concatenate_datasets(all_datasets)
    print(f"Total samples: {len(ds)}")

    # Print the first sample as a visual check
    print("\nFirst sample preview:")
    print(ds[0])

    # Download content for each sample
    print("\nDownloading content from S3...")
    ds = ds.map(download_contents, input_columns="blob_id", num_proc=NUM_PROC)

    # Filter out failed downloads
    ds = ds.filter(lambda x: x["download_success"])
    print(f"Successfully downloaded {len(ds)} samples")

    # Save to JSONL
    save_to_jsonl(ds, args.output)


if __name__ == "__main__":
    main()
