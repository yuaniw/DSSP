from setuptools import find_namespace_packages, setup

setup(
    name="dssp",
    packages=find_namespace_packages(include=["dssp", "dssp.*"]),
    package_data={"dssp": ["config/*.yaml", "config/task/*.yaml"]},
)
