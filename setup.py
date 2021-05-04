"""
Setup script to install vae4ptp as a Python package.
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
with open(join(dirname(__file__), 'vae4ptp/VERSION')) as version_file:
    version = version_file.read().strip()

# Run setup()
setup(
    name='vae4ptp',
    version=version,
    description='vae4ptp: VAEs for PT profiles',
    url='https://github.com/timothygebhard/vae4ptp',
    install_requires=[
        'h5py==3.2.1',
        'matplotlib==3.4.1',
        'numpy==1.20.2',
        'pandas==1.2.4',
        'scipy==1.6.3',
        'tqdm==4.60.0',
    ],
    extras_require={
        'develop': [
            'coverage==5.5',
            'flake8==3.9.1',
            'mypy==0.812',
            'pytest==6.2.3',
            'pytest-cov==2.11.1',
        ]
    },
    packages=find_packages(),
    zip_safe=False,
)
