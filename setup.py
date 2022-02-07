import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

requirements = []
with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setuptools.setup(
    name="mist_ion",
    version="0.4.2",
    author="lap1dem",
    author_email="vadym.bidula@gmail.com",
    description="Ionosphere modeling for the MIST project.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/lap1dem/mist_ion",
    project_urls={
        "Bug Tracker": "https://github.com/lap1dem/mist_ion/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.7,<3.10",
    install_requires=requirements,
)