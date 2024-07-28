from setuptools import setup, find_packages

setup(
    name="intelisys",
    version="0.2.1",  # Make sure this matches the version in intelisys/__init__.py
    packages=find_packages(),
    install_requires=[
        "openai",
        "litellm",
        "jinja2",
        "onepasswordconnectsdk",
    ],
    author="Mark Powers",
    author_email="mpoweru@lifsys.com",
    description="Provides intelligence/AI services for the Lifsys Enterprise",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/lifsys/intelisys",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
