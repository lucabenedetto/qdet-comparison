from setuptools import find_packages, setup

setup(
    name='qdet_utils',
    packages=find_packages(),
    version='0.1.0',
    description='Utils for performing QDET',
    author='Luca Benedetto',
    license='',
    install_requires=[
        "gensim==4.2.0",
        "nltk==3.7",
        "numpy==1.23.4",
        "pandas==1.5.1",
        "pyirt==0.3.4",
        "PyYAML==6.0",
        "scikit-learn==1.1.3",
        "scipy==1.9.3",
        "seaborn==0.12.1",
        "textstat==0.7.3",
        "matplotlib==3.6.2",
        "transformers==4.27.4",
        "datasets==2.11.0",
        "evaluate==0.4.0",
    ],
)
