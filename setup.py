from setuptools import setup, find_packages

setup(
    name="glance",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "gwpy",
        "pycbc",
        "astropy",
	"pesummary",
        "bilby",
        "lalsuite",
    ],
    description="A Python package for GLANCE",
    author="Aniruddha",
    author_email="aniruddha.chakraborty@tifr.res.in",
)
