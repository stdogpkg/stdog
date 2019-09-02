from setuptools import setup, find_packages

readme = open('README.md', 'r')
README_TEXT = readme.read()
readme.close()

setup(
    name="stdog",
    version="1.0.1",
    packages=find_packages(exclude=["build", ]),
    long_description=README_TEXT,
    install_requires=["tensorflow", "scipy", "numpy"],
    include_package_data=True,
    license="MIT",
    description="Structure and Dynamics on Graphs",
    author_email="messias.physics@gmail.com",
    author="Bruno Messias; Thomas K. Peron",
    download_url="https://github.com/stdogpkg/stdog/archive/v1.0.1.tar.gz", 
    keywords=[
        "gpu", "science", "complex-networks", "graphs", "dynamics",
         "tensorflow", "kuramoto"
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Topic :: Physics ::  Mathematics :: SCIENTIFIC/ENGINEERING",
    ],
    url="https://github.com/stdogpkg/stdog"
)
