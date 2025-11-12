from setuptools import setup

setup(
    name="visual_search_engine",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A generic framework for building visual search engines with pluggable backends.",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="http://your-repository-url.com",
    packages=['visual_search_engine'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        "numpy",
        "pillow",
    ],
)
