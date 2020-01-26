# Always prefer setuptools over distutils
from setuptools import setup, find_packages
from os import path

curr_dir = path.abspath(path.dirname(__file__))

MAJOR = 0
MINOR = 2
PATCH = 1
VERSION = "{}.{}.{}".format(MAJOR, MINOR, PATCH)

if __name__ == "__main__":

    # Get the long description from the README file
    with open(path.join(curr_dir, "README.rst"), encoding="utf-8") as file:
        long_description = file.read()

    setup(
        name="region",
        version=VERSION,
        description="Package offering regionalization algorithms",
        long_description=long_description,
        url="https://github.com/pysal/region",
        maintainer="PySAL Developers",
        maintainer_email="pysal-dev@googlegroups.com",
        license="BSD 3-clause",
        # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
        # 3 - Alpha; 4 - Beta; 5 - Production/Stable
        classifiers=[
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Science/Research",
            "Intended Audience :: Developers",
            "Intended Audience :: Education",
            "Topic :: Scientific/Engineering",
            "Topic :: Scientific/Engineering :: GIS",
            "License :: OSI Approved :: BSD License",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.5",
            "Programming Language :: Python :: 3.6",
        ],
        keywords="regionalization spatial clustering",
        packages=find_packages(exclude=["contrib", "doc", "test*"]),
        install_requires=[
            "geopandas",
            "libpysal",
            "networkx>=2.0.0",
            "numpy>=1.10.4",
            "pulp",
            "pytest",
            "scipy",
            "sklearn",
        ],
        python_requires=">3.4",
    )
