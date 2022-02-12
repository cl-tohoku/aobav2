from setuptools import setup


requirements = [name.rstrip() for name in open('requirements.txt')]
develop_requirements = [
    "mirai_translate==0.1.3",
    "googletrans==4.0.0rc1",
]

setup(
    name="aobav2",
    packages=["aobav2"],
    version="0.0.1",
    description="aobav2 bot",
    author="TohokuNLP",
    install_requires=requirements,
    extras_require={
        "develop": develop_reqiorements,
    },
)
