import os

def cleanup_broken_files():
    """
    Deletes specific broken PDF files from the pypapers directory.
    """
    pypapers_dir = 'pypapers'
    
    broken_files = [
        "Instance attack an explanation-based vulnerability analysis framework against DNNs for malware detection.pdf",
        "Computational Intelligence Approaches in Developing Cyberattack Detection System.pdf",
        "An Attribute Extraction for Automated Malware Attack Classification and Detection Using Soft Computing Techniques.pdf",
        "Android malware detection based on a hybrid deep learning model.pdf",
        "Semantic based greedy levy gradient boosting algorithm for phishing detection.pdf",
        "A Study on the Application of Distributed System Technology-Guided Machine Learning in Malware Detection.pdf",
        "Comparison of two-classification models based on neural network for DGA domain name detection.pdf",
        "Detecting malicious code variants using convolutional neural network (CNN) with transfer learning.pdf",
        "Towards Effective and Robust Neural Trojan Defenses via Input Filtering.pdf"
    ]

    if not os.path.isdir(pypapers_dir):
        print(f"Directory not found: {pypapers_dir}")
        return

    print("Starting cleanup of broken PDF files...")
    deleted_count = 0
    for filename in broken_files:
        file_path = os.path.join(pypapers_dir, filename)
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                print(f"Deleted: {filename}")
                deleted_count += 1
            except OSError as e:
                print(f"Error deleting {filename}: {e}")
        else:
            print(f"File not found, skipping: {filename}")

    print(f"\nCleanup complete. Deleted {deleted_count} files.")

if __name__ == "__main__":
    cleanup_broken_files()