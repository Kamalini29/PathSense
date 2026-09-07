from setuptools import setup, find_packages

setup(
    name="pathsense",
    version="1.0.0",
    description="Real-Time Monocular Depth & Object Detection Spatial Navigation Assistant",
    author="Antigravity Team",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "numpy",
        "opencv-python",
        "torch",
        "torchvision",
        "ultralytics",
        "transformers",
        "fastapi",
        "uvicorn",
        "websockets",
        "pyttsx3",
    ],
)
