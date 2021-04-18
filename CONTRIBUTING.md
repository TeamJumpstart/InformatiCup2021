# Contribution Guidelines

Improvements to the codebase and new features are welcome.

This project follows a strict feature branch workflow and maintains a linear git history.
Thus, _never push directly into `main`!_

Instead, to contribute:

- Create new branch with a meaningful name
- Commit you changes. Ensure your code is properly formatted (see below). Use unit tests where appropriate.
- Make sure your branch is not behind the `main` branch. Otherwise, rebase your branch on `main`.
- Create a pull request. Your code will be reviewed by a team member and either be merged or changes will be requested.

## Formatting

All Python code is formatted in Facebook style with 120 columns via `flake8` (see [setup.cfg](setup.cfg) for exact settings).

For VS code, settings are provided which enable autoformatting on saving.
