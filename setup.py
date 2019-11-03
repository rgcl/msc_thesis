# -*- coding: utf-8 -*-
# Copyright (C) 2019 Rodrigo González
# Licensed under the MIT licence - see LICENSE.txt
# Author: Rodrigo González-Castillo

from distutils.command.build import build

from setuptools import setup

setup(
    name='msc_thesis',
    version='0.1',
    packages=['thesis_code'],
    entry_points={
        'console_scripts': [
            'thesis-A = msc_thesis.stages.A_determining_filters.determining_filters:cli'
        ]
    },
    install_requires=[
        'numpy', 'matplotlib', 'astropy', 'pcigale', 'configobj', 'click'
    ],
    package_data={'data': ['msc_thesis/data/*']},
    author='Rodrigo González Castillo',
    author_email='rodrigo.gonzalez@uamail.cl',
    description='Msc thesis code: Determining the most important spectroscopic information for spectro-photometric '
                'modeling',
    license='MIT',
    keywords='astrophysics, galaxy sed'
)