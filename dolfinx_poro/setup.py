from setuptools import setup, find_packages

VERSION = "0.1.0"

REQUIREMENTS = ["fenics-dolfinx>=0.7.0", "numpy>=1.21.0"]

setup(
    name="dolfinx_poro",
    version=VERSION,
    description="A DOLFINx based frontend for poroelasticity",
    author="Maximilian Brodbeck",
    author_email="maximilian.brodbeck@isd.uni-stuttgart.de",
    python_requires=">3.7.0",
    packages=find_packages(),
    install_requires=REQUIREMENTS,
    zip_safe=False,
)
