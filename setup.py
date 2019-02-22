import distutils
import os

from setuptools import find_packages, setup
from setuptools.command import install

HERE = os.path.abspath(os.path.dirname(__file__))

def read(*parts):  # Stolen from txacme
    with open(os.path.join(HERE, *parts)) as f:
        return f.read()

setup(
    name='entity_embeddings_categorical',
    version='0.1',
    license='BSD-3-Clause',
    url='https://github.com/bresan/entity_embeddings_categorical',
    description='',
    long_description=read('README.md'),
    long_description_content_type="text/markdown",
    author='Rodrigo Bresan',
    author_email='',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3.6',
    ],
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'hello = app.__main__:main',
        ]
    }
)