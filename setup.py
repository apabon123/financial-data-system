from setuptools import setup, find_packages

setup(
    name="financial-data-system",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'pandas',
        'requests',
        'pyyaml',
        'duckdb',
        'python-dotenv',
        'python-dateutil'
    ]
) 