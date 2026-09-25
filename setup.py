#!/usr/bin/env python3

"""The setup script."""

from setuptools import find_packages, setup

with open('requirements.txt') as f:
    install_requires = [
        line.strip()
        for line in f
        if line.strip() and not line.lstrip().startswith('#')
    ]

with open('README.md') as f:
    long_description = f.read()


CLASSIFIERS = [
    'Development Status :: 4 - Beta',
    'License :: OSI Approved :: Apache Software License',
    'Operating System :: OS Independent',
    'Intended Audience :: Science/Research',
    'Programming Language :: Python',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.10',
    'Programming Language :: Python :: 3.11',
    'Programming Language :: Python :: 3.12',
    'Programming Language :: Python :: 3.13',
    'Programming Language :: Python :: 3.14',
    'Topic :: Scientific/Engineering',
]

setup(
    name='esp-lab',
    version='1.4.0',
    description='Diagnostic and analysis utilities for E3SM S2D ensemble predictions',
    long_description=long_description,
    long_description_content_type='text/markdown',
    python_requires='>=3.10',
    maintainer='ESP-Lab / E3SM Team',
    maintainer_email='tking@ucar.edu',
    classifiers=CLASSIFIERS,
    url='https://esp-lab.readthedocs.io',
    project_urls={
        'Documentation': 'https://esp-lab.readthedocs.io',
        'Source': 'https://github.com/zhangshixuan1987/ESP-Lab',
        'Tracker': 'https://github.com/zhangshixuan1987/ESP-Lab/issues',
    },
    packages=find_packages(exclude=('tests',)),
    include_package_data=True,
    install_requires=install_requires,
    license='Apache 2.0',
    zip_safe=False,
    entry_points={},
    keywords='ESP-Lab, E3SM, S2D, SMYLE, Earth System Predictions, diagnostics',
)
