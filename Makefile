.PHONY: release release-patch release-minor release-major book-build book-preview portal-build portal-preview site-build site-preview site-watch site-check showcase-sims showcase-video

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

site-build:
	@.venv/bin/python scripts/site_workflow.py stage

site-preview:
	@.venv/bin/python scripts/site_workflow.py preview --port $(or $(PORT),8080)

site-watch:
	@.venv/bin/python scripts/site_workflow.py preview --watch --port $(or $(PORT),8080)

site-check:
	@.venv/bin/python scripts/site_workflow.py check

showcase-sims:
	@.venv/bin/python scripts/generate_docs_showcase.py --skip-video

showcase-video:
	@if [ -z "$(VIDEO)" ]; then \
		echo "Usage: make showcase-video VIDEO=/path/to/tracked_video.mp4"; \
		exit 1; \
	fi
	@.venv/bin/python scripts/generate_docs_showcase.py --video "$(VIDEO)" --skip-simulations
