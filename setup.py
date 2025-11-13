from setuptools import setup

setup(
    name="cl_ml_tools",
    version="0.1.0",
    author="Ananda Sarangaram",
    author_email="asarangaram@gmail.com",
    description="A machine learning toolset for embedded / edge devices.",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="http://anandas.in",
    packages=['cl_ml_tools'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        "numpy",
    ],
)
