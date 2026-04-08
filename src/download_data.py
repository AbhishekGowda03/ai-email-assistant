import urllib.request
import zipfile
import os

def download_spamassassin():
    """Download SpamAssassin dataset"""
    url = "https://spamassassin.apache.org/old/publiccorpus/20050311_spam_2.tar.bz2"
    ham_url = "https://spamassassin.apache.org/old/publiccorpus/20030228_easy_ham.tar.bz2"
    
    print("Downloading spam dataset...")
    urllib.request.urlretrieve(url, "data/raw/spam.tar.bz2")
    
    print("Downloading ham dataset...")
    urllib.request.urlretrieve(ham_url, "data/raw/ham.tar.bz2")
    
    print("Download complete!")

if __name__ == "__main__":
    download_spamassassin()