from setuptools import find_packages, setup

setup(
    author="Drew Blasius",
    author_email="drewblasius@gmail.com",
    name="stock_cutting",
    version="0.1.0",
    packages=find_packages(include=["stock_cutting", "stock_cutting.*"]),
    license="MIT License",
    long_description=open("README.rst").read(),
)
