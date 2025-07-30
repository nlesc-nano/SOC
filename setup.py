from setuptools import setup, find_packages

setup(
    name="soc",
    version="0.1.0",
    description="Spin-Orbit Coupling tools with Python/C++ backend",
    author="Ivan Infante",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    include_package_data=True,
    install_requires=[
        "numpy>=1.22",
        "scipy>=1.8",
        "matplotlib>=3.5",
        "pymatgen>=2022.0",
    ],
    entry_points={
        "console_scripts": [
            "soc = soc.soc_driver:main",  # CLI entry: soc command runs soc.soc_driver.main()
        ],
    },
    python_requires=">=3.9",
)
