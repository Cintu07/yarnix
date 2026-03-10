import urllib.request
import os

def download_data():
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    filepath = "C:\\Users\\kolag\\Desktop\\round\\Yarnix\\tinyshakespeare.txt"
    
    print(f"Downloading TinyShakespeare dataset to {filepath}...")
    try:
        urllib.request.urlretrieve(url, filepath)
        filesize = os.path.getsize(filepath) / (1024 * 1024)
        print(f"Download complete! Size: {filesize:.2f} MB")
        
        # Verify it can be read
        with open(filepath, 'r', encoding='utf-8') as f:
            sample = f.read(100)
            print(f"Sample data:\n---\n{sample}\n---")
            
    except Exception as e:
        print(f"Failed to download: {e}")

if __name__ == "__main__":
    download_data()
