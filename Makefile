.PHONY: release release-patch release-minor release-major

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
