DIR_GITHUB_ACTIONS := .github

install_requirements_testing:
	pip install -r $(DIR_GITHUB_ACTIONS)/requirements_testing
	pip install .
