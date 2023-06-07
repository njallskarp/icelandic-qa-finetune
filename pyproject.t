[build-system]
requires = ["setuptools", "wheel"]

[tool.poetry]
name = "my-awesome-project"
version = "0.1.0"
description = "An example of a Python project"
authors = ["Your Name <your.name@example.com>"]

[tool.poetry.dependencies]
python = "^3.7"
torch = "^1.9"
transformers = "^4.8"

[tool.poetry.dev-dependencies]
pytest = "^6.2"
