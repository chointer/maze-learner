from setuptools import setup, find_packages

setup(
    name="gym-maze",
    version="0.1.1",
    author="chointer",
    author_email="chointer04@gmail.com",
    description="A gymnaisum maze environment",
    url="https://github.com/chointer/maze",
    packages=find_packages(),
    python_requires='>=3.11',
    install_requires=[
        "gymnasium>=0.29.1",
        "matplotlib>=3.9.0",
        "numpy>=2.0.0",
        "pygame>=2.6.0",
    ],
)