import sys
from pathlib import Path
from setuptools import setup, find_packages

if sys.version_info < (3,):
    sys.exit('do_qtl requires Python >= 3.6')

try:
    from do_qtl import __author__, __email__
except ImportError:  # Deps not yet installed
    __author__ = __email__ = ''


setup(
    name='do_qtl',
    version='0.1',
    description='Genetic association analyses in Diversity Outbred mice',
    long_description=Path('README.md').read_text('utf-8'),
    long_description_content_type="text/markdown",
    url='http://github.com/calico/do_qtl',
    download_url='https://github.com/calico/do_qtl/archive/0.1.tar.gz',
    author=__author__,
    author_email=__email__,
    license='Apache',
    python_requires='>=3.6',
    install_requires=[
        l.strip() for l in
        Path('requirements.txt').read_text('utf-8').splitlines()
        ],
    packages=find_packages(),
    classifiers=[
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'Topic :: Genetics',
    ],
)
