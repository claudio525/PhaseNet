from setuptools import setup, find_packages

setup(
    name="phase_net",
    version="1.0",
    packages=find_packages(),
    url="https://github.com/claudio525/PhaseNet",
    description="Slight modification to PhaseNet for easier usage, https://github.com/wayneweiqiang/PhaseNet",
    install_requires=["numpy", "pandas", "scipy", "tensorflow>=2.1.0"],
    scripts=[],
    package_data={"phase_net": ["model/*"]},
    include_package_data=True
)
