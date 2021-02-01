#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Note: To use the 'upload' functionality of this file, you must:
#   $ pip install twine

import io
import os
import sys
from shutil import rmtree

from setuptools import find_packages, setup, Command

# Package meta-data.
NAME = 'diego'
DESCRIPTION = 'Diego: Data IntElliGence Out.'
URL = 'https://github.com/lai-bluejay/diego'
EMAIL = 'lai.bluejay@gmail.com'
AUTHOR = 'Charles Lai'
REQUIRES_PYTHON = '>=3.6.0'
VERSION = '0.2.6.1'


# What packages are required for this module to be executed?
REQUIRED = [
    # 'requests', 'maya', 'records',
    'numpy>=1.16.2',
    'scipy>=0.19.0',
    'scikit-learn>=0.23',
    'deap>=1.0',
    'update_checker>=0.16',
    'tqdm>=4.26.0',
    'stopit>=1.1.1',
    'pandas>=1.0',
    'xgboost',
    'pyrfr>=0.7,<0.9',
    'distributed',
    'dask',
    'smac>=0.8',
    'ConfigSpace<0.5,>=0.4.14',
    'auto-sklearn>=0.11',
    'liac-arff',
    'sklearn-contrib-lightning'
    
]

# What packages are optional?
EXTRAS = {
    # 'fancy feature': ['django'],
}
here = os.path.abspath(os.path.dirname(__file__))

def parse_requirements(REQUIRED, here):
    with open(os.path.join(here, 'requirements.txt')) as fp:
        install_reqs = [r.rstrip() for r in fp.readlines()
                        if not r.startswith('#') and not r.startswith('git+')]
    new_reqs = [r for r in install_reqs if r not in REQUIRED]
    REQUIRED += new_reqs
    return REQUIRED
REQUIRED = parse_requirements(REQUIRED, here)

# The rest you shouldn't have to touch too much :)
# ------------------------------------------------
# Except, perhaps the License and Trove Classifiers!
# If you do change the License, remember to change the Trove Classifier for that!

here = os.path.abspath(os.path.dirname(__file__))

# setup_reqs = ['Cython', 'numpy']
# with open(os.path.join(here, 'requirements.txt')) as fp:
#     install_reqs = [r.rstrip() for r in fp.readlines()
#                     if not r.startswith('#') and not r.startswith('git+')]


# Import the README and use it as the long-description.
# Note: this will only work if 'README.md' is present in your MANIFEST.in file!
try:
    with io.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
        long_description = '\n' + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

# Load the package's __version__.py module as a dictionary.
about = {}
if not VERSION:
    project_slug = NAME.lower().replace("-", "_").replace(" ", "_")
    with open(os.path.join(here, project_slug, '__version__.py')) as f:
        exec(f.read(), about)
else:
    about['__version__'] = VERSION


class UploadCommand(Command):
    """Support setup.py upload."""

    description = 'Build and publish the package.'
    user_options = []

    @staticmethod
    def status(s):
        """Prints things in bold."""
        print('\033[1m{0}\033[0m'.format(s))

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        try:
            self.status('Removing previous builds…')
            rmtree(os.path.join(here, 'dist'))
        except OSError:
            pass

        self.status('Building Source and Wheel (universal) distribution…')
        os.system('{0} setup.py sdist bdist_wheel --universal'.format(sys.executable))

        self.status('Uploading the package to PyPI via Twine…')
        os.system('twine upload dist/*')

        self.status('Pushing git tags…')
        os.system('git tag v{0}'.format(about['__version__']))
        os.system('git push --tags')
        
        sys.exit()


# Where the magic happens:
setup(
    name=NAME,
    version=about['__version__'],
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type='text/markdown',
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(exclude=('tests',)),
    # If your package is a single module, use this instead of 'packages':
    # py_modules=['mypackage'],

    # entry_points={
    #     'console_scripts': ['mycli=mymodule:cli'],
    # },
    install_requires=REQUIRED,
    extras_require=EXTRAS,
    include_package_data=True,
    license='MIT',
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: Implementation :: CPython',
        'Programming Language :: Python :: Implementation :: PyPy'
    ],
    # $ setup.py publish support.
    cmdclass={
        'upload': UploadCommand,
    },
)