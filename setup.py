from setuptools import setup, find_packages

setup(
    name='space',
    version='1.0',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'gensim==4.3.2',
        'h5py==3.12.1',
        'loguru==0.7.3',
        'matplotlib==3.10.0',
        'numba==0.60.0',
        'numpy==1.26.4',
        'pandas==2.2.3',
        'pecanpy==2.0.9',
        'scikit_learn==1.6.0',
        'tqdm==4.67.1',
        'transformers==4.47.1',
        "scipy==1.11.4",
        "tensorboard==2.18.0",
        "umap-learn==0.5.7",
        "seaborn==0.13.2",
        "cafa-eval==1.2.1"
    ],
)
