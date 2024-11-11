from setuptools import setup, find_packages

setup(
    name='space',
    version='1.0',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'gensim==4.3.2',
        'h5py',
        'loguru',
        'matplotlib',
        'numba',
        'numpy',
        'pandas',
        'pecanpy',
        'scikit_learn',
        'tqdm',
        'transformers',
    ],
)
