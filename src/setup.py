from setuptools import setup, find_packages

setup(
    name="pycellmech",
    version="2.1.0",
    author="Janan Arslan",
    author_email="janan.arslan@icm-institute.org",
    description="A shape-based feature extractor for medical and biological studies",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/icm-dac/pycellmech",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: CC BY-NC License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        matplotlib
        numpy
        math
        opencv-python
        pandas
        scikit-learn
    ],
    include_package_data=True,
)
