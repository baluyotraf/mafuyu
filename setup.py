from setuptools import setup, find_packages

setup(
    name='mafuyu',
    version='0.0.1',
    packages=find_packages(exclude=['tests.*', 'tests']),
    install_requires=[
        'numpy>=1.14.5',
        'matplotlib>=2.2.2',
        'scikit-learn>=0.19.2',
    ],
)
