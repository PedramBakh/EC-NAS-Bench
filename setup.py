from pathlib import Path
from setuptools import setup, find_packages

def read_reqs(path="requirements.txt"):
    if Path(path).exists():
        reqs = []
        for line in Path(path).read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                reqs.append(line)
        return reqs
    return []

setup(
    name="ecnas",
    version="0.1.0",
    description="EC-NAS",
    packages=find_packages(),
    install_requires=read_reqs(),
    python_requires=">=3.9",
)
