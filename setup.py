"""
Setup script to install ml4ptp as a Python package.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from os.path import join, dirname
from setuptools import find_packages, setup


# -----------------------------------------------------------------------------
# RUN setup() FUNCTION
# -----------------------------------------------------------------------------

# Get version from VERSION file
with open(join(dirname(__file__), 'ml4ptp/VERSION')) as version_file:
    version = version_file.read().strip()

# Run setup()
setup(
    name='ml4ptp',
    version=version,
    description='ml4ptp: machine learning for PT profiles',
    url='https://github.com/timothygebhard/ml4ptp',
    install_requires=[
        'h5py==3.6.0',
        'matplotlib==3.5.1',
        'numpy==1.22.2',
        'pandas==1.4.1',
        'pyyaml==6.0',
        'rich==11.2.0',
        'scikit-learn==1.0.2',
        'scipy==1.8.0',
        'torch==1.10.2',
        'tqdm==4.62.3',
    ],
    extras_require={
        'develop': [
            'coverage',
            'flake8',
            'mypy',
            'pylint',
            'pytest',
            'pytest-cov',
        ]
    },
    packages=find_packages(),
    zip_safe=False,
)
