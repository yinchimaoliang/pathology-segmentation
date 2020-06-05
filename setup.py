from setuptools import find_packages, setup

setup(
    name='pathseg',
    packages=find_packages(exclude=('tests', 'docs', 'images')))
