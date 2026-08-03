.PHONY: bootstrap bootstrap-scaffold verify verify-env lint typecheck test check

bootstrap:
	pwsh -NoProfile -ExecutionPolicy Bypass -File scripts/dev/bootstrap.ps1

bootstrap-scaffold:
	pwsh -NoProfile -ExecutionPolicy Bypass -File scripts/dev/bootstrap.ps1 -ScaffoldOnly

verify-env:
	python scripts/dev/verify_environment.py

verify:
	python scripts/dev/verify_repository.py

verify-clean:
	python scripts/dev/verify_repository.py --require-clean

lint:
	python -m ruff check .

typecheck:
	python -m mypy src

test:
	python -m pytest

check: verify-env verify lint typecheck test
