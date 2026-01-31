from pathlib import Path

from setuptools import setup


README = Path(__file__).with_name("README.md").read_text(encoding="utf-8")

setup(
    name="sequenzanalyse",
    version="0.1.0",
    description="Sequential analysis pipeline in the German tradition of Objektive Hermeneutik.",
    long_description=README,
    long_description_content_type="text/markdown",
    author="",
    license="",
    python_requires=">=3.10",
    install_requires=["openai"],
    packages=["sequenzanalyse"],
    package_dir={"sequenzanalyse": "."},
    include_package_data=True,
    package_data={"sequenzanalyse": ["_prompts/*.txt"]},
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
