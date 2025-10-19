# Annolid GUI Decomposition Plan

## Current State
- `annolid/gui/app.py` still coordinates logging, flag persistence, realtime widgets, and Dino feature tools in addition to canvas orchestration.
- Menu and action creation now route through `MenuController`, and tracking lifecycle lives in `TrackingController`, but other subsystems remain inside `AnnolidWindow`.
- Several widgets (e.g., `FlagTableWidget`, patch similarity dialogs) depend on mutable window attributes without clear ownership boundaries.

## Guiding Principles
- Continue grouping code by responsibility: entry/bootstrap, controllers (stateful feature coordinators), services (pure logic), and widgets (presentation).
- Controllers should receive narrow window/canvas facades instead of accessing all attributes directly.
- Preserve behaviour while extracting modules; prefer shims/adapters and incremental migrations.
- Document controller entry points so new features extend the appropriate layer.

## Target Module Map
1. **`annolid/gui/cli.py` + `application.py`** – CLI parsing and Qt bootstrap helpers.
2. **Controllers**
   - `tracking.py` – tracking workers, segment editors, status updates. *(Done)*
   - `menu.py` – action descriptors, tool/menubar wiring. *(Initial pass complete)*
   - `flags.py` *(planned)* – pinned flag persistence, YAML updates, table widget signalling.
   - `dino.py` *(planned)* – patch similarity, PCA map requests, settings persistence.
3. **Services** – pure logic extracted from controllers (e.g., YAML IO helpers, screenshot saving).
4. **Widgets** – remain focused on rendering and emit signals consumed by controllers.

## Milestones
### Milestone A – Bootstrap split & hygiene
- [x] Extract CLI/config parsing into `annolid/gui/cli.py`.
- [x] Add `create_qapp(argv)` helper returning `QApplication`.
- [x] Make `main(argv=None)` delegate to helpers; enable unit tests for CLI parsing without Qt.

### Milestone B – Tracking subsystem
- [x] Inventory all tracking-related methods in `AnnolidWindow`.
- [x] Introduce `TrackingController` accepting signal callbacks and minimal window interface.
- [x] Migrate tracking dialog interactions and worker wiring into the controller; expose narrow methods (`start_tracking`, etc.).

### Milestone C – Menu & action builder
- [x] Move menu construction and toolbar population out of `__init__` via `MenuController`.
- [ ] Define declarative action descriptors (text, shortcut, icon, slot) and build menus from structured data.
- [ ] Centralize shortcut registration and enable opt-in grouping by feature (annotation, conversion, analytics).

### Milestone D – Ancillary controllers
- [x] Port pinned flag persistence & YAML updates to `FlagsController`.
- [x] Migrate patch similarity & PCA services into dedicated controller with clear lifecycle hooks (polish & testing still pending).
- [ ] Validate each controller with focused tests (Qt’s `QSignalSpy` where practical).

### Milestone E – Documentation & migration
- [ ] Document controller responsibilities and integration flow in developer docs.
- [ ] Update contributor guidelines to encourage controller/service additions instead of expanding `AnnolidWindow`.

## Immediate Next Actions
1. Add tests (skipped when Qt unavailable) covering controller behaviours (`FlagsController`, `DinoController`) and AI-model lazy bootstrapping.
2. Polish `DinoController` (extend declarative config, ensure overlays support masks, and finalise menu integration) before extracting additional services.
3. Document controller integration patterns and update coding guidelines so new GUI code uses controllers/services instead of `AnnolidWindow` internals.
