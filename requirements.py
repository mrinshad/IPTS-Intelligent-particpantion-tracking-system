import subprocess
import sys

def install_required_packages():
    packages = [
        "opencv-python",
        "numpy",
        "Pillow",
        "flask",
        "flask-mysqldb",
        "dlib",
        "face-recognition",
        "csv"
    ]

    for package in packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"Successfully installed {package}")
        except subprocess.CalledProcessError as e:
            print(f"Failed to install {package}: {e}")

# Call the function to install the packages
install_required_packages()
