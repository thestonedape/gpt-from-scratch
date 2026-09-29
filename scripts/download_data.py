import requests
from pathlib import Path
import argparse
from tqdm import tqdm


def download_shakespeare():
    """Download Shakespeare's complete works as sample data"""
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    output_path = Path("data/raw/shakespeare.txt")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading Shakespeare dataset...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    
    with open(output_path, 'wb') as f, tqdm(
        total=total_size,
        unit='B',
        unit_scale=True,
        desc="Shakespeare"
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            pbar.update(len(chunk))
    
    print(f"Downloaded to {output_path}")
    return str(output_path)


def download_openwebtext():
    """Download OpenWebText subset (~8GB uncompressed)"""
    print("OpenWebText is large (~8GB). Downloading subset...")
    url = "https://huggingface.co/datasets/Skylion007/openwebtext/resolve/main/subsets/urlsf_subset00-0_data.tar"
    output_path = Path("data/raw/openwebtext.tar")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"This will take a while. Download OpenWebText subset...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    
    with open(output_path, 'wb') as f, tqdm(
        total=total_size,
        unit='B',
        unit_scale=True,
        desc="OpenWebText"
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            pbar.update(len(chunk))
    
    print(f"Downloaded to {output_path}")
    print("Extract with: tar -xf data/raw/openwebtext.tar -C data/raw/")
    return str(output_path)


def download_wikitext():
    """Download WikiText-103 (~180MB)"""
    url = "https://s3.amazonaws.com/research.metamind.io.s3-website-us-west-2.amazonaws.com/wikitext/wikitext-103-raw-v1.zip"
    output_path = Path("data/raw/wikitext103.zip")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading WikiText-103...")
    print("Note: If this fails, manually download from:")
    print("https://blog.einstein.ai/the-wikitext-long-term-dependency-language-modeling-dataset/")
    
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(output_path, 'wb') as f, tqdm(
            total=total_size,
            unit='B',
            unit_scale=True,
            desc="WikiText-103"
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))
        
        print(f"Downloaded to {output_path}")
        
        import zipfile
        print("Extracting...")
        with zipfile.ZipFile(output_path, 'r') as zip_ref:
            zip_ref.extractall("data/raw/")
        
        print("✓ Done! Training file: data/raw/wikitext-103-raw/wiki.train.raw")
        return "data/raw/wikitext-103-raw/wiki.train.raw"
    except Exception as e:
        print(f"\n✗ Download failed: {e}")
        print("\nAlternative: Use the books dataset you already have,")
        print("or train longer (20-30 epochs) with current data")
        return None


def download_pg19():
    """Download multiple books from Project Gutenberg"""
    book_ids = [
        (11, "alice_in_wonderland"),
        (84, "frankenstein"),
        (1342, "pride_and_prejudice"),
        (2701, "moby_dick"),
        (98, "tale_of_two_cities"),
        (1661, "sherlock_holmes"),
        (174, "dorian_gray"),
        (1260, "jane_eyre"),
        (844, "importance_of_being_earnest"),
        (16, "peter_pan"),
        (1952, "yellow_wallpaper"),
        (1399, "anna_karenina"),
        (730, "oliver_twist"),
        (46, "christmas_carol"),
        (219, "heart_of_darkness"),
        (158, "emma"),
        (1232, "prince_and_pauper"),
        (345, "dracula"),
        (768, "wuthering_heights"),
        (2554, "crime_and_punishment")
    ]
    
    output_dir = Path("data/raw/books")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_text = []
    
    for book_id, title in tqdm(book_ids, desc="Downloading books"):
        try:
            url = f"https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt"
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            book_path = output_dir / f"{title}.txt"
            with open(book_path, 'wb') as f:
                f.write(response.content)
            
            all_text.append(response.text)
        except Exception as e:
            print(f"Failed to download {title}: {e}")
    
    combined_path = Path("data/raw/books_large.txt")
    with open(combined_path, 'w', encoding='utf-8') as f:
        f.write('\n\n'.join(all_text))
    
    print(f"\n✓ Downloaded {len(all_text)} books")
    print(f"✓ Combined file: {combined_path}")
    print(f"✓ Size: {len(''.join(all_text)):,} characters (~{len(''.join(all_text))//1_000_000}MB)")
    return str(combined_path)


def download_wikipedia_sample():
    """Download Wikipedia sample from HuggingFace"""
    from datasets import load_dataset
    
    print("Downloading Wikipedia sample...")
    output_path = Path("data/raw/wikipedia_sample.txt")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        dataset = load_dataset("wikipedia", "20220301.en", split="train[:10000]", trust_remote_code=True)
        
        print(f"Processing {len(dataset)} articles...")
        texts = [article['text'] for article in tqdm(dataset, desc="Processing")]
        
        combined = '\n\n'.join(texts)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(combined)
        
        print(f"\n✓ Saved to {output_path}")
        print(f"✓ Size: {len(combined):,} characters")
        return str(output_path)
    except Exception as e:
        print(f"✗ Failed: {e}")
        print("Install with: pip install datasets")
        return None


def download_gutenberg_book(book_id: int, title: str):
    """Download a book from Project Gutenberg"""
    url = f"https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt"
    output_path = Path(f"data/raw/{title}.txt")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading {title}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    
    with open(output_path, 'wb') as f, tqdm(
        total=total_size,
        unit='B',
        unit_scale=True,
        desc=title
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            pbar.update(len(chunk))
    
    print(f"Downloaded to {output_path}")
    return str(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download datasets for training")
    parser.add_argument(
        "--dataset",
        type=str,
        default="shakespeare",
        choices=["shakespeare", "books", "books_large", "wikipedia", "wikitext", "openwebtext"],
        help="Dataset: shakespeare (1MB), books (6MB), books_large (20MB), wikipedia (50MB sample), wikitext (500MB), openwebtext (8GB)"
    )
    args = parser.parse_args()
    
    print(f"\nDataset sizes:")
    print("  shakespeare: ~1MB (testing only)")
    print("  books: ~6MB (10 novels)")
    print("  books_large: ~20MB (20 classic novels) ← RECOMMENDED")
    print("  wikipedia: ~50MB (10k Wikipedia articles, requires 'datasets' lib)")
    print("  wikitext: ~500MB (currently broken)")
    print("  openwebtext: ~8GB (realistic, very large)")
    print()
    
    if args.dataset == "shakespeare":
        print("⚠️  Warning: Shakespeare is tiny, only for testing!")
        download_shakespeare()
    elif args.dataset == "books":
        download_pg19()
    elif args.dataset == "books_large":
        print("Downloading 20 classic novels (~20MB)...")
        download_pg19()
    elif args.dataset == "wikipedia":
        download_wikipedia_sample()
    elif args.dataset == "wikitext":
        download_wikitext()
    elif args.dataset == "openwebtext":
        download_openwebtext()
    
    print("\nNext steps:")
    print("1. Train tokenizer: python tokenizer/train_tokenizer.py --input data/raw/*.txt --output data/tokenizer/tokenizer.json")
    print("2. Preprocess data: python scripts/preprocess.py --input data/raw/*.txt --output_dir data/processed --tokenizer data/tokenizer/tokenizer.json")
    print("3. Train model: python train.py")
