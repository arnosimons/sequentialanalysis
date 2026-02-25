from pathlib import Path
from setuptools import setup

README = Path(__file__).with_name("README.md").read_text(encoding="utf-8")

setup(
    name="sequentialanalysis",
    version="0.1.0",
    description="Sequential analysis pipeline in the German tradition of Objektive Hermeneutik.",
    long_description=README,
    long_description_content_type="text/markdown",
    python_requires=">=3.10",
    install_requires=[
        "openai",
        "pydantic>=2",  # openai_model.py imports pydantic
    ],
    packages=[
        "sequentialanalysis",
        "sequentialanalysis.llms",
    ],
    package_dir={"sequentialanalysis": "."},
    include_package_data=True,
    package_data={"sequentialanalysis": ["_prompts/*.txt"]},
)
