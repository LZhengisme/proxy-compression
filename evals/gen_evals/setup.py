from setuptools import setup

with open("README.md") as readme_file:
    readme = readme_file.read()

requirements = []

setup(
    name='gen_eval',
    description="A framework for the evaluation of code generation models.",
    long_description=readme,
    license="Apache 2.0",
    packages=["gen_eval"],
    install_requires=requirements,
)
