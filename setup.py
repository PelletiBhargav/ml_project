from setuptools import find_packages,setup
from typing import List

hyp_e_bot='-e .'
def get_req(file_path:str)->List[str]:
    '''
    
    this function will return the list req
    
    '''
    with open(file_path) as f:
        req = f.readlines()
        req = [r.strip() for r in req if r.strip() and r.strip() != '-e .']
    return req

setup(
    name="mlproject",
    version="0.0.1",
    author="Bhargav",
    author_email="pelletivenkatabhargav03@gmail.com",
    packages=find_packages(),
    install_requires=get_req('req.txt')

    
)