# Changelog

All notable changes to the AgentLaboratory project will be documented in this file.

At the beginning, we had to install:
PyPDF2
flask
flask_sqlalchemy
sentence-transformers
tf-keras

## [Unreleased]

### Added
- Added `save_loc` parameter to PaperSolver class to properly handle file saving locations
- Enhanced `compile_latex` function to automatically create the tex directory if it doesn't exist

### Fixed
- Fixed initialization of PaperSolver in `ai_lab_repo.py` to correctly pass the lab directory for file saving
- Updated PaperReplace and PaperEdit classes to properly use the save_loc parameter

## [1.0.0] - 2024-03-25

- Initial release 