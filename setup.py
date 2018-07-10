from setuptools import setup, find_packages

readme = open('README','r')
README_TEXT = readme.read()
readme.close()

setup(
    name="tkuramoto",
    version="0.0.1",
    packages = find_packages(exclude=["build",]),
    scripts=["tkuramoto/bin/tkuramoto"],
    long_description = README_TEXT,
    install_requires=["tensorflow", "scipy", "numpy"],
    include_package_data=True,
    license="AGPLv3",
    description="Dynamics in large networks",
    author="Bruno Messias",
    author_email="messias.physics@gmail.com",
    download_url="https://github.com/devmessias/phd_tkuramoto/archive/0.0.1.tar.gz",
    keywords=["gpu", "science","complex-networks", "dynamics"],

    classifiers=[
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: GNU Affero General Public License v3 or later (AGPLv3+)",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python",
        "Topic :: Text Processing :: Markup :: LaTeX",
    ],
    url="https://github.com/devmessias/phd_tkuramoto"
)
