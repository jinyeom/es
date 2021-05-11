from setuptools import setup, find_packages


setup(
    name="es",
    description="Simple implementation of OpenAI Evolution Strategy",
    version="0.0.1",
    author="Jin Yeom",
    install_requires=["numpy>=1.20"],
    packages=find_packages(),
)
