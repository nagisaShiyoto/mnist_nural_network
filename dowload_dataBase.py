import requests
import os

def download_mnist():
    # Create directory
    os.makedirs('mnist_data', exist_ok=True)
    
    # Google Storage base URL
    base_url = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
    
    # Files to download
    files = [
        'train-images-idx3-ubyte.gz',
        'train-labels-idx1-ubyte.gz',
        't10k-images-idx3-ubyte.gz',
        't10k-labels-idx1-ubyte.gz'
    ]
    
    for filename in files:
        filepath = os.path.join('mnist_data', filename)
        if not os.path.exists(filepath):
            url = base_url + filename
            print(f"Downloading {filename}...")
            
            try:
                response = requests.get(url)
                response.raise_for_status()  # Check if download was successful
                
                with open(filepath, 'wb') as f:
                    f.write(response.content)
                print(f"Successfully downloaded {filename}")
                
            except Exception as e:
                print(f"Error downloading {filename}: {e}")
                # If file was partially downloaded, remove it
                if os.path.exists(filepath):
                    os.remove(filepath)

if __name__ == "__main__":
    try:
        import requests
    except ImportError:
        print("Installing requests library...")
        import subprocess
        subprocess.check_call(['pip', 'install', 'requests'])
        
    download_mnist()