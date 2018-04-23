# based on https://github.com/pypa/sampleproject/blob/master/setup.py and
# https://github.com/pysal/pysal/blob/master/setup.py

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
# To use a consistent encoding
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

MAJOR = 0
MINOR = 1
PATCH = 5
VERSION = "{}.{}.{}".format(MAJOR, MINOR, PATCH)

if __name__ == "__main__":
    # Get the long description from the README file
    with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
        long_description = f.read()

    setup(
        name='region',

        version=VERSION,

        description='Package offering regionalization algorithms',
        long_description=long_description,

        url='https://github.com/pysal/region',

        maintainer="PySAL Developers",
        maintainer_email='pysal-dev@googlegroups.com',

        license='BSD 3-clause',

        # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
        classifiers=[
            #   3 - Alpha
            #   4 - Beta
            #   5 - Production/Stable
            'Development Status :: 3 - Alpha',

            # Indicate who your project is intended for
            'Intended Audience :: Science/Research',
            'Intended Audience :: Developers',
            'Intended Audience :: Education',
            'Topic :: Scientific/Engineering',
            'Topic :: Scientific/Engineering :: GIS',

            # Pick your license as you wish (should match "license" above)
            'License :: OSI Approved :: BSD License',

            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.5',
            'Programming Language :: Python :: 3.6',
        ],

        # What does your project relate to?
        keywords='regionalization spatial clustering',

        # You can just specify the packages manually here if your project is
        # simple. Or you can use find_packages().
        packages=find_packages(exclude=['contrib', 'doc', 'test*']),

        # See: https://packaging.python.org/en/latest/requirements.html
        install_requires=[
            'geopandas',
            'libpysal',
            'networkx<2.0.0',
            'numpy>=1.10.4',
            'pulp',
            'pytest',
            'scipy',
            'sklearn',
            'pysal'
        ],

        python_requires='~=3.3',  # ~=3.3 means >=3.3 and <4
    )
