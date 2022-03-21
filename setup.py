from os import path
from setuptools import setup, find_packages
import sys
import versioneer
import glob


# NOTE: This file must remain Python 2 compatible for the foreseeable future,
# to ensure that we error out properly for people with outdated setuptools
# and/or pip.


here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md'), encoding='utf-8') as readme_file:
    readme = readme_file.read()

with open(path.join(here, 'requirements.txt')) as requirements_file:
    # Parse requirements.txt, ignoring any commented-out lines.
    requirements = [line for line in requirements_file.read().splitlines()
                    if not line.startswith('#')]


setup(
    name='denoising_autoencoder',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description="denoising XPCS with AI",
    long_description=readme,
    author="Tatiana Konstantinova",
    author_email='tkonstant@bnl.gov',
    url='https://github.com/bnl/automating_XPCS_analysis_pub',
    python_requires='>=3.6',
    packages=find_packages(exclude=['docs', 'tests']),
    include_package_data=True,
    #package_data={
    #    'denoising': ['denoising/models/*']
    #},
    install_requires=requirements,
    license="BSD (3-clause)",
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
    ],
)
