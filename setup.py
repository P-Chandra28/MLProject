from setuptools import find_packages,setup
from typing import List

def get_req(file_path:str)->List[str]:
    E_DOT="-e ."
    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace("\n","") for req in requirements]
        if E_DOT in requirements:
            requirements.remove(E_DOT)
    return requirements

setup(
    name="MLProject",
    version="0.0.1",
    author="ChandraSekhar",
    author_email="chandu22sekhar@gmail.com",
    packages=find_packages(),
    install_requires=get_req('requirements.txt')
)