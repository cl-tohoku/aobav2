from setuptools import setup


requirements = [name.rstrip() for name in open('requirements.txt')]

setup(
    name="aobav2",
    packages=["aobav2"],
    version="0.0.1",
    description="aobav2 bot",
    author="TohokuNLP",
    install_requires=requirements,
    extras_require={
        "develop": []
    },
)
