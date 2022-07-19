from setuptools import find_packages, setup

author = "benedikt fuchs"
author_email = "benedikt.fuchs.staw@hotmail.com"
name = "ner-eval-dashboard"
url = "https://github.com/helpmefindaname/ner-eval-dashboard"
python_name = name.replace("-", "_")

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = f.readlines()

with open("requirements-dev.txt", "r", encoding="utf-8") as f:
    dev_requriements = f.readlines()

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()
    long_description = long_description[long_description.index("#") :]


def extract_short_description() -> str:
    # extract the short description which is the short sentence after the header in the readme
    start = long_description.index("\n")
    if "##" in long_description:
        end = long_description.index("##")
    else:
        end = len(long_description)
    return long_description[start:end].strip()


setup(
    name=name,
    version="0.0.1",
    description=extract_short_description(),
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="MIT License",
    license_files=("LICENSE",),
    author=author,
    author_email=author_email,
    url=url,
    install_requires=requirements,
    packages=find_packages(exclude="tests"),
    extras_require={"dev": dev_requriements},
    entry_points={"console_scripts": [f"{python_name} = {python_name}.cli:main"]},
    setuptools_git_versioning={
        "enabled": True,
    },
    setup_requires=["setuptools-git-versioning"],
)
