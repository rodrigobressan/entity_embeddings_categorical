import os

from setuptools import find_packages, setup

HERE = os.path.abspath(os.path.dirname(__file__))


def read_file(*file_name: str) -> str:
    """
    This function is used to read a given file and return its contents
    :param file_name the file to be read
    :return: the string containing the file contents
    """
    with open(os.path.join(HERE, *file_name)) as f:
        return f.read()


setup(
    name='entity_embeddings_categorical',
    version='0.3',
    license='BSD-3-Clause',
    url='https://github.com/bresan/entity_embeddings_categorical',
    description='',
    long_description=read_file('README.md'),
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
