from setuptools import setup

setup(
    # pgeg
    name = 'pgeg',

    # Version number:
    version = '0.1.0',

    # Application author details:
    author = 'Kent Kawashima',
    author_email = 'i@kentwait.com',

    # Package
    packages = ['pgeg'],
    url = 'http://pypi.python.org/pypi/pgeg',
    license = 'LINCESE',
    author='Kent Kawashima',
    description = 'Python objects and fucntions for population and evolutionary genetics analysis'

    # Required packages
    requires = [
        'numpy',
        'pandas',
        'bioseq',
    ]
)
