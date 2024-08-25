from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="intelisys",
    version="0.5.5",  # Incremented version number
    author="Lifsys Enterprise",
    author_email="contact@lifsys.com",
    description="Intelligence/AI services for the Lifsys Enterprise with enhanced max_history_words, efficient history trimming, and improved async support",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/lifsys/intelisys",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.7",
    install_requires=[
        "openai>=1.0.0",
        "jinja2>=3.0.0",
        "onepasswordconnectsdk>=1.0.0",
        "anthropic>=0.3.0",
        "pillow>=8.0.0",
        "termcolor>=1.1.0",
        "aiohttp>=3.7.4",  # Added for improved async support
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "flake8>=3.9",
            "black>=21.5b1",
        ],
    },
    project_urls={
        "Documentation": "https://intelisys.readthedocs.io/",
        "Source": "https://github.com/lifsys/intelisys",
        "Tracker": "https://github.com/lifsys/intelisys/issues",
    },
)