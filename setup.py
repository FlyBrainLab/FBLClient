from setuptools import setup
from setuptools import find_packages

install_requires=[
      'neuroballad',
      'txaio',
      'h5py',
      'autobahn',
      'twisted[tls]',
      'autobahn_sync',
      'jupyter',
      'seaborn',
      'requests',
      'dataclasses'
]

setup(name='FlyBrainLab',
      version='0.1.0',
      description='Main Client of the FlyBrainLab Project',
      author='Mehmet Kerem Turkcan',
      author_email='mkt2126@columbia.edu',
      url='',
      install_requires=install_requires,
      download_url='',
      license='BSD-3-Clause',
      packages=find_packages())
