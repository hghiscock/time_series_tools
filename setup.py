from setuptools import find_packages, setup

setup(
    name="time_series_tools",
    packages=find_packages(),
    include_package_data=True,
    author="",
    license="",
    install_requires=[
        "numpy==1.24.2",
        "PyWavelets==1.4.1",
    ]
)
