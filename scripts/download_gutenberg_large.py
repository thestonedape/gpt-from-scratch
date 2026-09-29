import requests
from pathlib import Path
from tqdm import tqdm
import time

def download_many_gutenberg_books():
    """Download 200 popular books from Project Gutenberg"""
    
    # Top 200 popular book IDs from Project Gutenberg
    book_ids = [
        1342, 84, 1080, 11, 1661, 2701, 46, 98, 1952, 74,
        768, 174, 345, 1260, 16, 120, 205, 730, 1399, 2554,
        844, 219, 158, 2600, 135, 1232, 1497, 766, 2852, 1400,
        5200, 2591, 514, 1184, 64317, 244, 6130, 23, 1727, 100,
        161, 1259, 2542, 58585, 786, 408, 35, 209, 829, 113,
        132, 43, 203, 996, 1250, 2148, 41, 76, 27780, 1404,
        105, 215, 375, 308, 45, 394, 28054, 147, 55, 80,
        37106, 815, 1998, 1251, 514, 55, 863, 50133, 10676, 140,
        160, 67979, 4363, 1695, 2600, 2097, 4280, 30254, 2542, 142,
        153, 1400, 1322, 1257, 25344, 521, 2413, 158, 145, 27827,
        8800, 2814, 1298, 26184, 1934, 541, 2265, 829, 37134, 375,
        36, 1400, 514, 1250, 54, 1250, 99, 1400, 2348, 244,
        # Add more IDs (continuing to 200)
        514, 120, 98, 1952, 74, 345, 829, 786, 514, 140,
        160, 161, 203, 205, 215, 219, 244, 308, 345, 375,
        394, 408, 514, 521, 541, 730, 766, 768, 786, 815,
        829, 863, 996, 1080, 1184, 1232, 1250, 1251, 1257, 1259,
        1260, 1298, 1322, 1342, 1399, 1400, 1404, 1497, 1661, 1695,
        1727, 1934, 1952, 1998, 2097, 2148, 2265, 2348, 2413, 2542,
        2554, 2591, 2600, 2701, 2814, 2852, 4280, 4363, 5200, 6130,
        8800, 10676, 25344, 26184, 27780, 27827, 28054, 30254, 36, 37106, 37134, 50133, 54, 55, 58585, 64317, 67979
    ]
    
    # Remove duplicates
    book_ids = list(set(book_ids))[:200]
    
    output_dir = Path("data/raw/gutenberg_large")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_text = []
    successful = 0
    
    print(f"Downloading {len(book_ids)} books from Project Gutenberg...")
    print("This will take 5-10 minutes...\n")
    
    for book_id in tqdm(book_ids, desc="Downloading"):
        try:
            # Try different URL formats
            urls = [
                f"https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt",
                f"https://www.gutenberg.org/files/{book_id}/{book_id}.txt",
                f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"
            ]
            
            text = None
            for url in urls:
                try:
                    response = requests.get(url, timeout=10)
                    if response.status_code == 200:
                        text = response.text
                        break
                except:
                    continue
            
            if text:
                book_path = output_dir / f"book_{book_id}.txt"
                with open(book_path, 'w', encoding='utf-8') as f:
                    f.write(text)
                all_text.append(text)
                successful += 1
            
            time.sleep(0.5)  # Be nice to Project Gutenberg servers
            
        except Exception as e:
            continue
    
    combined_path = Path("data/raw/gutenberg_mega.txt")
    print(f"\nCombining {successful} books...")
    with open(combined_path, 'w', encoding='utf-8') as f:
        f.write('\n\n'.join(all_text))
    
    total_chars = len(''.join(all_text))
    print(f"\n✓ Successfully downloaded {successful} books")
    print(f"✓ Combined file: {combined_path}")
    print(f"✓ Size: {total_chars:,} characters (~{total_chars//1_000_000}MB)")
    print(f"✓ Estimated tokens: ~{total_chars//4:,}")
    return str(combined_path)

if __name__ == "__main__":
    download_many_gutenberg_books()
