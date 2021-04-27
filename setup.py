from setuptools import setup
from setuptools import find_packages

install_requires = [
    "txaio",
    "h5py",
    "autobahn",
    "twisted[tls]",
    "autobahn_sync @ git+https://github.com/mkturkcan/autobahn-sync",
    "jupyter",
    "pyOpenSSL",
    "seaborn",
    "requests",
    "dataclasses; python_version<'3.7'",
    "msgpack",
    "msgpack-numpy",
    "neuroballad @ git+https://github.com/flybrainlab/neuroballad",
    "service_identity",
    "crochet",
    "matplotlib",
    "fastcluster",
    "networkx",
    "pandas",
    "scipy",
    "sympy",
    "nose",
    "jupyterlab>=2.2.8, <=3.0.10",
    "pywin32; platform_system=='Windows'"
]

extras_require_utilities = {
        "gem @ git+https://github.com/palash1992/GEM.git",
        "graspy<=0.1.1",
        "umap-learn",
        "tensorflow",
        "graphviz",
        "sklearn",
        "cdlib",
        "pyclustering",
        "nxcontrol @ git+https://github.com/mkturkcan/nxcontrol"
    }

extras_require_full = extras_require_utilities

setup(
    name="FlyBrainLab",
    version="1.1.4",
    description="Main Client of the FlyBrainLab Project",
    author="Mehmet Kerem Turkcan",
    author_email="mkt2126@columbia.edu",
    url="https://flybrainlab.fruitflybrain.org",
    download_url="",
    license="BSD-3-Clause",
    install_requires=install_requires,
    extras_require = {
        "full": extras_require_full,
        "utilities": extras_require_utilities,
    },
    dependency_links=["https://github.com/flybrainlab/neuroballad/tarball/master#egg=neuroballad-0.1.0",
                      "https://github.com/mkturkcan/autobahn-sync/tarball/master#egg=autobahn_sync-0.3.2",
                      "https://github.com/palash1992/GEM/tarball/master#egg=gem-1.0.0",
                      "https://github.com/mkturkcan/nxcontrol/tarball/master#egg=nxcontrol-0.2"],
    packages=find_packages(),
)
