import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pytorch-constraints",
    version="0.0.1",
    author="todo",
    author_email="todo",
    description="Constraints for PyTorch Models",
    long_description=long_description,
    url="https://github.com/pytorch-constraints/pytorch-constraints",
    # package_dir={"": "pytorch_constraints"},
    packages=setuptools.find_packages(),
)
