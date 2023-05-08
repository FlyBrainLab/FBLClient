from setuptools import setup
from setuptools import find_packages
from os import path

install_requires = [
    "numpy",
    "txaio",
    "h5py",
    "autobahn",
    "twisted[tls]",
    "autobahn_sync @ git+https://github.com/mkturkcan/autobahn-sync",
    "pyOpenSSL",
    "seaborn",
    "requests",
    "dataclasses; python_version<'3.7'",
    "msgpack > 1.0",
    "msgpack-numpy",
    "neuroballad @ git+https://github.com/flybrainlab/neuroballad",
    "matplotlib",
    "networkx",
    "pandas",
    "ipython", # no longer need if remove get_slow_connectivity
    "pywin32; platform_system=='Windows'",
    "graphviz",
    "jupyterlab >= 3.0, <3.6", # 3.6 has some issue for
]

extras_require_utilities = {
        "nxt_gem<=2.0.1",
        "graspy<=0.1.1",
        "scipy",
        "umap-learn",
        "tensorflow",
        "scikit-learn",
        "cdlib",
        "pyclustering",
        "nxcontrol @ git+https://github.com/mkturkcan/nxcontrol"
    }

extras_require_full = extras_require_utilities

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()
    
setup(
    name="FlyBrainLab",
    version="1.1.10",
    description="FlyBrainLab User-side Backend",
    author="Mehmet Kerem Turkcan, Yiyin Zhou",
    author_email="mkt2126@columbia.edu, yiyin@ee.columbia.edu",
    url="https://flybrainlab.fruitflybrain.org",
    download_url="",
    license="BSD-3-Clause",
    long_description=long_description,
    long_description_content_type='text/markdown',
    python_requires=">=3.6",
    install_requires=install_requires,
    extras_require = {
        "full": extras_require_full,
        "utilities": extras_require_utilities,
    },
    packages=find_packages(),
)
