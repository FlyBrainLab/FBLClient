from setuptools import setup
from setuptools import find_packages

install_requires = [
    "txaio",
    "h5py",
    "autobahn",
    "twisted[tls]",
    "autobahn_sync",
    "jupyter",
    "seaborn",
    "requests",
    "dataclasses",
    'msgpack',
    'msgpack-numpy'
]

setup(
    name="FlyBrainLab",
    version="1.0",
    description="Main Client of the FlyBrainLab Project",
    author="Mehmet Kerem Turkcan",
    author_email="mkt2126@columbia.edu",
    url="https://flybrainlab.fruitflybrain.org",
    install_requires=install_requires,
    download_url="",
    license="BSD-3-Clause",
    packages=find_packages(),
)
