.PHONY: release release-patch release-minor release-major book-build book-preview portal-build portal-preview

release:
	@if [ -z "$(VERSION)" ]; then \
		echo "Usage: make release VERSION=X.Y.Z [PUSH=1] [SKIP_CHECKS=1]"; \
		exit 1; \
	fi
	@args=""; \
	if [ "$(PUSH)" = "1" ]; then args="$$args --push"; fi; \
	if [ "$(SKIP_CHECKS)" = "1" ]; then args="$$args --skip-checks"; fi; \
	./scripts/release.sh "$(VERSION)" $$args

release-patch:
	@args=""; \
	if [ "$(PUSH)" = "1" ]; then args="$$args --push"; fi; \
	if [ "$(SKIP_CHECKS)" = "1" ]; then args="$$args --skip-checks"; fi; \
	./scripts/release.sh patch $$args

release-minor:
	@args=""; \
	if [ "$(PUSH)" = "1" ]; then args="$$args --push"; fi; \
	if [ "$(SKIP_CHECKS)" = "1" ]; then args="$$args --skip-checks"; fi; \
	./scripts/release.sh minor $$args

release-major:
	@args=""; \
	if [ "$(PUSH)" = "1" ]; then args="$$args --push"; fi; \
	if [ "$(SKIP_CHECKS)" = "1" ]; then args="$$args --skip-checks"; fi; \
	./scripts/release.sh major $$args

book-build:
	@source .venv/bin/activate && jupyter-book build book

book-preview: book-build
	@open book/_build/html/index.html

portal-build:
	@source .venv/bin/activate && mkdocs build --clean --config-file mkdocs.yml

portal-preview:
	@source .venv/bin/activate && mkdocs serve --config-file mkdocs.yml
