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
    "dataclasses",
    'msgpack',
    'msgpack-numpy',
    'neuroballad @ git+https://github.com/flybrainlab/neuroballad',
    'service_identity',
    'crochet',
    'matplotlib',
    'fastcluster',
    'networkx',
    'pandas',
    'scipy',
    'sympy',
    'nose',
    'jupyterlab>=2.2.8',
    'pywin32; platform_system=="Windows"'
]

setup(
    name="FlyBrainLab",
    version="1.1.3",
    description="Main Client of the FlyBrainLab Project",
    author="Mehmet Kerem Turkcan",
    author_email="mkt2126@columbia.edu",
    url="https://flybrainlab.fruitflybrain.org",
    download_url="",
    license="BSD-3-Clause",
    install_requires=install_requires,
    dependency_links=['https://github.com/flybrainlab/neuroballad/tarball/master#egg=neuroballad-0.1.0',
                      'https://github.com/mkturkcan/autobahn-sync/tarball/master#egg=autobahn_sync-0.3.2'],
    packages=find_packages(),
)
