import os
import sys

from setuptools import setup, find_packages

import CryFold


setup(
    name="CryFold",
    author='Baoquan Su',
    author_email='202217127mail.sdu.edu.cn',
    entry_points={
        "console_scripts": [
            "build = CryFold.build:main",
        ],
    },
    packages=find_packages(),
    package_data={
        '': ['./*.json', './utils/*.py','./utils/*.txt' ,'./Unet/*.py', './CryNet/*.py', './checkpoint/*.pth'],
    },
    include_package_data=True,
    version=CryFold.__version__,
)
