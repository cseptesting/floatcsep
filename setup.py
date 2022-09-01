from setuptools import setup, find_packages


setup(
    name='fecsep',
    version='0.1.0',
    author='Pablo Iturrieta',
    author_email='pciturri@gfz-potsdam.de',
    license='LICENSE',
    description='fecsep',
    packages=find_packages(),
    include_package_data=False,
    python_requires=">=3.8",
    url='git@git.gfz-potsdam.de:csep-group/fecsep-quadtree.git'
)
