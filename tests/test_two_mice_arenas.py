"""Deterministic Node.js validation for neuroscience behavioral arenas (no WebGL/CDN needed)."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


def _find_node_binary() -> str | None:
    node = shutil.which("node")
    if node is not None:
        return node
    try:
        from playwright._impl._driver import compute_driver_executable

        driver = compute_driver_executable()
        candidate = Path(driver[0] if isinstance(driver, tuple) else driver)
        if candidate.is_file():
            return str(candidate)
    except Exception:
        return None
    return None


def _run_js(body: str):
    node = _find_node_binary()
    if node is None:
        pytest.skip("Node.js is required to validate the JavaScript arena module")
    result = subprocess.run(
        [node, "--input-type=module", "-e", body],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
        timeout=20,
    )
    return json.loads(result.stdout)


def test_arena_catalog_ids_aliases_and_error_handling():
    result = _run_js(
        """
        import {
          ARENA_CATALOG,
          ARENA_IDS,
          ARENA_LABELS,
          DEFAULT_ARENA_ID,
          resolveArenaId,
          getArenaSpec,
          summarizeArenaSpec,
        } from './annolid/gui/assets/threejs/two_mice_arenas.js';

        const aliasChecks = {
          epm: resolveArenaId('epm'),
          nor: resolveArenaId('nor'),
          lightDark: resolveArenaId('light-dark'),
          threeChamber: resolveArenaId('3_chamber'),
          yMaze: resolveArenaId('y-maze'),
          tMaze: resolveArenaId('t-maze'),
          barnes: resolveArenaId('barnes'),
          circular: resolveArenaId('circular'),
          byLabel: resolveArenaId('Elevated Plus Maze'),
          defaultFallback: resolveArenaId('unknown_xyz', { allowDefault: true }),
        };
        let invalidError = null;
        try {
          resolveArenaId('nonexistent_arena');
        } catch (err) {
          invalidError = err.name;
        }
        let emptyError = null;
        try {
          resolveArenaId('');
        } catch (err) {
          emptyError = err.name;
        }
        console.log(JSON.stringify({
          count: ARENA_CATALOG.length,
          ids: ARENA_IDS,
          labels: ARENA_LABELS,
          defaultId: DEFAULT_ARENA_ID,
          aliasChecks,
          invalidError,
          emptyError,
          summaryKeys: Object.keys(summarizeArenaSpec(getArenaSpec('elevated_plus_maze'))).sort(),
        }));
        """
    )
    assert result["count"] == 10
    assert result["defaultId"] == "open_field"
    assert set(result["ids"]) == {
        "open_field",
        "home_cage",
        "circular_open_field",
        "elevated_plus_maze",
        "y_maze",
        "t_maze",
        "three_chamber",
        "light_dark_box",
        "barnes_maze",
        "novel_object",
    }
    assert result["labels"][0] == "Open Field"
    assert result["labels"][1] == "Home Cage (Bedding)"
    assert result["aliasChecks"] == {
        "epm": "elevated_plus_maze",
        "nor": "novel_object",
        "lightDark": "light_dark_box",
        "threeChamber": "three_chamber",
        "yMaze": "y_maze",
        "tMaze": "t_maze",
        "barnes": "barnes_maze",
        "circular": "circular_open_field",
        "byLabel": "elevated_plus_maze",
        "defaultFallback": "open_field",
    }
    assert result["invalidError"] == "RangeError"
    assert result["emptyError"] == "RangeError"
    assert result["summaryKeys"] == [
        "bounds",
        "category",
        "description",
        "hudTitle",
        "id",
        "label",
        "zones",
    ]


def test_every_arena_spawns_fly_zones_and_waypoints_are_strictly_navigable():
    result = _run_js(
        """
        import {
          ARENA_CATALOG,
          arenaSignedDistance,
          isPointInArena,
          sampleArenaSpawnPose,
        } from './annolid/gui/assets/threejs/two_mice_arenas.js';
        import { createSeededRandom } from './annolid/gui/assets/threejs/two_mice_simulation.js';

        const rng = createSeededRandom(2026);
        const failures = [];
        for (const arena of ARENA_CATALOG) {
          for (let i = 0; i < arena.spawnSlots.length; i++) {
            const [sx, sz] = arena.spawnSlots[i].position;
            const d = arenaSignedDistance(arena, sx, sz);
            if (d < 2.0) {
              failures.push(`${arena.id}:slot[${i}] clearance=${d.toFixed(3)}`);
            }
          }
          const { reward, hazard } = arena.flyZones;
          if (!isPointInArena(arena, reward.x, reward.z, 1.8)) {
            failures.push(`${arena.id}:reward out of bounds`);
          }
          if (!isPointInArena(arena, hazard.x, hazard.z, 1.8)) {
            failures.push(`${arena.id}:hazard out of bounds`);
          }
          for (const wp of arena.navigationWaypoints) {
            if (!isPointInArena(arena, wp.x, wp.z, 1.5)) {
              failures.push(`${arena.id}:waypoint(${wp.x},${wp.z}) out of bounds`);
            }
          }
          // Repeated random spawn sampling must remain inside walkable bounds
          const placed = [];
          for (let k = 0; k < 6; k++) {
            const pose = sampleArenaSpawnPose(arena, rng, placed);
            if (!isPointInArena(arena, pose.position[0], pose.position[1], 1.8)) {
              failures.push(`${arena.id}:sampled spawn ${k} invalid`);
            }
            placed.push(pose.position);
          }
        }
        console.log(JSON.stringify({ failures }));
        """
    )
    assert result["failures"] == []


def test_arena_collision_and_tail_containment_hold_under_adversarial_motion():
    result = _run_js(
        """
        import {
          ARENA_CATALOG,
          arenaSignedDistance,
          resolveArenaPosition,
          clampTailPointToArena,
          computeTailArenaLookahead,
          isTailSegmentClearOfArena,
        } from './annolid/gui/assets/threejs/two_mice_arenas.js';
        import { createSeededRandom } from './annolid/gui/assets/threejs/two_mice_simulation.js';

        const rng = createSeededRandom(777);
        const report = {};

        for (const arena of ARENA_CATALOG) {
          let x = arena.spawnSlots[0].position[0];
          let z = arena.spawnSlots[0].position[1];
          let vx = 4.0;
          let vz = -3.0;
          let wallFollowDir = 1;
          let minSignedDist = Infinity;
          let tailOutsideCount = 0;
          let nanDetected = false;

          for (let step = 0; step < 300; step++) {
            const angle = rng() * Math.PI * 2;
            const hx = Math.cos(angle);
            const hz = Math.sin(angle);
            // Push aggressively toward walls, doorways, obstacles, and reflex corners
            vx = vx * 0.75 + hx * 6.5;
            vz = vz * 0.75 + hz * 6.5;
            x += vx * 0.05;
            z += vz * 0.05;

            const res = resolveArenaPosition(
              arena, x, z, vx, vz, hx, hz, 1.0, 1.0, 1.0, 0.0, 0.0, 9.2, wallFollowDir
            );
            x = res.x;
            z = res.z;
            vx = res.vx;
            vz = res.vz;
            wallFollowDir = res.wallFollowDir;

            if (!Number.isFinite(x) || !Number.isFinite(z) || !Number.isFinite(vx) || !Number.isFinite(vz)) {
              nanDetected = true;
            }
            const sd = arenaSignedDistance(arena, x, z);
            minSignedDist = Math.min(minSignedDist, sd);

            // Tail clamp test around mouse position (including outside probes)
            const probeX = x + (rng() - 0.5) * 10.0;
            const probeZ = z + (rng() - 0.5) * 10.0;
            const clamped = clampTailPointToArena(arena, probeX, probeZ, 0.14);
            if (arenaSignedDistance(arena, clamped.x, clamped.z) < 0.08) {
              tailOutsideCount++;
            }
            // Lookahead must return finite direction
            const look = computeTailArenaLookahead(arena, x, z, hx, hz, 0.14, 1.4, hx, hz, 1);
            if (!Number.isFinite(look.safeX) || !Number.isFinite(look.safeZ)) {
              nanDetected = true;
            }
            // Segment check at valid center must pass for small zero-length step
            if (!isTailSegmentClearOfArena(arena, x, z, x, z, 0.14)) {
              tailOutsideCount++;
            }
          }

          report[arena.id] = {
            minSignedDist: Number(minSignedDist.toFixed(3)),
            tailOutsideCount,
            nanDetected,
          };
        }
        console.log(JSON.stringify(report));
        """
    )
    for arena_id, metrics in result.items():
        assert metrics["nanDetected"] is False, f"NaN in {arena_id}"
        assert metrics["minSignedDist"] >= 1.1, (
            f"Mouse breached boundary/obstacle in {arena_id}: {metrics}"
        )
        assert metrics["tailOutsideCount"] == 0, (
            f"Tail clamp failed in {arena_id}: {metrics}"
        )


def test_doorway_passage_and_unwalled_cliff_rearing_contracts():
    result = _run_js(
        """
        import {
          getArenaSpec,
          arenaSignedDistance,
          getArenaWallSupportPoint,
          classifyArenaZone,
        } from './annolid/gui/assets/threejs/two_mice_arenas.js';

        // 1. Three-Chamber and Light-Dark doorways have ample clearance for a mouse walking through
        const threeChamber = getArenaSpec('three_chamber');
        const lightDark = getArenaSpec('light_dark_box');
        const leftDoorClear = arenaSignedDistance(threeChamber, -7.0, 0.0);
        const rightDoorClear = arenaSignedDistance(threeChamber, 7.0, 0.0);
        const ldDoorClear = arenaSignedDistance(lightDark, -5.4, 0.0);

        // 2. Elevated Plus Maze: closed arms support wall rearing; open arm drop-offs return null
        const epm = getArenaSpec('elevated_plus_maze');
        const closedArmSupport = getArenaWallSupportPoint(epm, 11.0, 2.8, 0.0, 1.0);
        const openArmCliffSupport = getArenaWallSupportPoint(epm, 2.8, -11.0, 1.0, 0.0);

        // 3. Barnes Maze: unwalled elevated table never offers wall-supported rearing
        const barnes = getArenaSpec('barnes_maze');
        const barnesCliffSupport = getArenaWallSupportPoint(barnes, 13.0, 0.0, 1.0, 0.0);

        // 4. Zone classification spot checks across arenas
        const zones = {
          epmCenter: classifyArenaZone(epm, 0, 0),
          epmClosedW: classifyArenaZone(epm, -10, 0),
          epmClosedE: classifyArenaZone(epm, 10, 0),
          epmOpenN: classifyArenaZone(epm, 0, -10),
          epmOpenS: classifyArenaZone(epm, 0, 10),
          yHub: classifyArenaZone(getArenaSpec('y_maze'), 0, 0),
          yA: classifyArenaZone(getArenaSpec('y_maze'), 0, -12),
          yB: classifyArenaZone(getArenaSpec('y_maze'), 10, 6),
          yC: classifyArenaZone(getArenaSpec('y_maze'), -10, 6),
          tStart: classifyArenaZone(getArenaSpec('t_maze'), 0, 10),
          tChoice: classifyArenaZone(getArenaSpec('t_maze'), 0, -8.6),
          tLeft: classifyArenaZone(getArenaSpec('t_maze'), -12, -8.6),
          tRight: classifyArenaZone(getArenaSpec('t_maze'), 12, -8.6),
          tcSocial: classifyArenaZone(threeChamber, -14.2, 4.0),
          tcCenter: classifyArenaZone(threeChamber, 0, 0),
          tcObject: classifyArenaZone(threeChamber, 14.2, 4.0),
          ldDark: classifyArenaZone(lightDark, -12, 0),
          ldDoor: classifyArenaZone(lightDark, -5.4, 0),
          ldLight: classifyArenaZone(lightDark, 8, 0),
          barnesEscape: classifyArenaZone(barnes, 0, -11),
          norFamiliar: classifyArenaZone(getArenaSpec('novel_object'), -6.4, 1.5),
          norNovel: classifyArenaZone(getArenaSpec('novel_object'), 6.4, 1.5),
        };

        console.log(JSON.stringify({
          leftDoorClear: Number(leftDoorClear.toFixed(3)),
          rightDoorClear: Number(rightDoorClear.toFixed(3)),
          ldDoorClear: Number(ldDoorClear.toFixed(3)),
          closedArmHasSupport: Boolean(closedArmSupport),
          openArmCliffIsNull: openArmCliffSupport === null,
          barnesCliffIsNull: barnesCliffSupport === null,
          zones,
        }));
        """
    )
    assert result["leftDoorClear"] >= 3.5
    assert result["rightDoorClear"] >= 3.5
    assert result["ldDoorClear"] >= 3.5
    assert result["closedArmHasSupport"] is True
    assert result["openArmCliffIsNull"] is True
    assert result["barnesCliffIsNull"] is True
    assert result["zones"] == {
        "epmCenter": "center",
        "epmClosedW": "closed_arm_west",
        "epmClosedE": "closed_arm_east",
        "epmOpenN": "open_arm_north",
        "epmOpenS": "open_arm_south",
        "yHub": "center_hub",
        "yA": "arm_a",
        "yB": "arm_b",
        "yC": "arm_c",
        "tStart": "start_box",
        "tChoice": "choice_point",
        "tLeft": "left_arm",
        "tRight": "right_arm",
        "tcSocial": "social_cup_zone",
        "tcCenter": "center_chamber",
        "tcObject": "object_cup_zone",
        "ldDark": "dark_compartment",
        "ldDoor": "doorway",
        "ldLight": "light_compartment",
        "barnesEscape": "escape_quadrant",
        "norFamiliar": "familiar_object_zone",
        "norNovel": "novel_object_zone",
    }


def test_all_custom_arena_threejs_visuals_build_cleanly_in_node():
    result = _run_js(
        """
        import { THREE } from './annolid/gui/assets/threejs/annolid_threejs_runtime.js';
        import {
          ARENA_CATALOG,
          createArenaVisual,
        } from './annolid/gui/assets/threejs/two_mice_arenas.js';

        globalThis.document = {
          createElement() {
            return {
              width: 0,
              height: 0,
              getContext() {
                return new Proxy({}, {
                  get(_t, key) {
                    if (key === 'createRadialGradient' || key === 'createLinearGradient') {
                      return () => ({ addColorStop() {} });
                    }
                    return () => {};
                  },
                });
              },
            };
          },
        };

        const built = {};
        for (const spec of ARENA_CATALOG) {
          if (spec.legacyRect) continue;
          const group = createArenaVisual(THREE, spec, { simulationSeed: 20260710 });
          let meshCount = 0;
          let nonFiniteVertices = 0;
          group.traverse(node => {
            if (node.isMesh) {
              meshCount++;
              const pos = node.geometry?.attributes?.position;
              if (pos) {
                for (let i = 0; i < pos.count; i++) {
                  if (!Number.isFinite(pos.getX(i)) || !Number.isFinite(pos.getY(i)) || !Number.isFinite(pos.getZ(i))) {
                    nonFiniteVertices++;
                  }
                }
              }
            }
          });
          const box = new THREE.Box3().setFromObject(group);
          built[spec.id] = {
            meshCount,
            nonFiniteVertices,
            finiteBounds: Number.isFinite(box.min.x) && Number.isFinite(box.max.x),
          };
        }
        console.log(JSON.stringify(built));
        """
    )
    assert len(result) == 8
    for arena_id, info in result.items():
        assert info["meshCount"] >= 8, f"Expected rich 3D apparatus for {arena_id}"
        assert info["nonFiniteVertices"] == 0, f"Non-finite vertices in {arena_id}"
        assert info["finiteBounds"] is True, f"Invalid bounding box in {arena_id}"


def test_corridor_and_doorway_routing_avoids_interior_walls_and_obstacles():
    result = _run_js(
        """
        import {
          getArenaSpec,
          isArenaSegmentClear,
          computeArenaNavigationRoute,
          computeArenaWallEscapeTarget,
        } from './annolid/gui/assets/threejs/two_mice_arenas.js';

        const cases = [
          { id: 't_maze', from: [0, 8.2], to: [12.6, -8.6], expectRouted: true },
          { id: 'y_maze', from: [-10.6, 6.1], to: [0, -12.2], expectRouted: true },
          { id: 'elevated_plus_maze', from: [0, 11.8], to: [11.8, 0], expectRouted: true },
          { id: 'three_chamber', from: [14.2, 5.8], to: [-14.2, 5.8], expectRouted: true },
          { id: 'light_dark_box', from: [10.2, 7.5], to: [-11.6, 7.5], expectRouted: true },
          { id: 'novel_object', from: [-6.4, -8.6], to: [-6.4, 3.4], expectRouted: true },
          { id: 'open_field', from: [-6.0, 4.0], to: [6.8, -4.8], expectRouted: false },
        ];

        const checks = {};
        for (const c of cases) {
          const arena = getArenaSpec(c.id);
          const route = computeArenaNavigationRoute(arena, c.from[0], c.from[1], c.to[0], c.to[1]);
          let allHopsClear = true;
          for (let i = 0; i < route.path.length - 1; i++) {
            const a = route.path[i];
            const b = route.path[i + 1];
            if (!isArenaSegmentClear(arena, a.x, a.z, b.x, b.z, 0.85)) {
              allHopsClear = false;
            }
          }
          checks[c.id] = {
            routed: route.routed,
            expectRouted: c.expectRouted,
            hops: route.path.length - 1,
            allHopsClear,
            firstHop: [Number(route.x.toFixed(2)), Number(route.z.toFixed(2))],
            effectiveDist: Number(route.effectiveDist.toFixed(2)),
          };
        }

        const tEscape = computeArenaWallEscapeTarget(getArenaSpec('t_maze'), -16.0, -8.6);
        const ldEscape = computeArenaWallEscapeTarget(getArenaSpec('light_dark_box'), -15.0, 8.0);
        console.log(JSON.stringify({ checks, tEscape, ldEscape }));
        """
    )
    for arena_id, info in result["checks"].items():
        assert info["routed"] is info["expectRouted"], (
            f"Routing mismatch for {arena_id}: {info}"
        )
        assert info["allHopsClear"] is True, (
            f"Route clipped wall/obstacle in {arena_id}: {info}"
        )
        if info["expectRouted"]:
            assert info["hops"] >= 2
    # T-maze choice point is at (0, -8.6), NOT (0, 0) outside the crossbar
    assert result["tEscape"]["x"] == pytest.approx(0.0, abs=0.1)
    assert result["tEscape"]["z"] == pytest.approx(-8.6, abs=0.1)
    # Light-Dark escape from dark corner routes cleanly toward dark center / doorway
    assert result["ldEscape"]["z"] == pytest.approx(0.0, abs=0.1)


def test_fly_brain_connectome_navigates_all_multi_arm_and_chamber_mazes_to_reward():
    result = _run_js(
        """
        import {
          getArenaSpec,
          arenaSignedDistance,
          resolveArenaPosition,
          computeArenaNavigationRoute,
          computeArenaWallEscapeTarget,
          computeArenaWhiskerClearance,
          getArenaFlyZones,
        } from './annolid/gui/assets/threejs/two_mice_arenas.js';
        import { FlyBrainMouseController } from './annolid/gui/assets/threejs/two_mice_fly_brain.js';

        function runClosedLoop(arenaId, startPos, startYaw, maxSteps = 720) {
          const arena = getArenaSpec(arenaId);
          const zones = getArenaFlyZones(arena);
          const brain = new FlyBrainMouseController({ seed: 20260719, dynamics: 'spiking' });
          let x = startPos[0];
          let z = startPos[1];
          let yaw = startYaw;
          let vx = 0;
          let vz = 0;
          let wallFollowDir = 1;
          let reachedReward = false;
          let minDistToReward = Infinity;
          let minWallClearance = Infinity;
          const dt = 1 / 30;

          for (let step = 0; step < maxSteps; step++) {
            const fx = Math.sin(yaw);
            const fz = Math.cos(yaw);
            const lx = Math.cos(yaw);
            const lz = -Math.sin(yaw);

            const project = (tx, tz) => {
              const dx = tx - x;
              const dz = tz - z;
              const dist = Math.hypot(dx, dz);
              const inv = 1 / Math.max(dist, 1e-4);
              return { dist, fwd: (dx * fx + dz * fz) * inv, left: (dx * lx + dz * lz) * inv };
            };

            const reward = zones.reward;
            const directDist = Math.hypot(reward.x - x, reward.z - z);
            minDistToReward = Math.min(minDistToReward, directDist);
            if (directDist < reward.radius) {
              reachedReward = true;
              break;
            }

            const route = computeArenaNavigationRoute(arena, x, z, reward.x, reward.z);
            const rProj = project(route.x, route.z);
            const decay = route.routed ? 18.0 : 8.5;
            const prox = Math.exp(-Math.max(0, route.effectiveDist - reward.radius * 0.65) / decay);
            const plume = prox * 0.85;
            let odorLeft = 0;
            let odorRight = 0;
            if (rProj.left >= 0) {
              odorLeft += plume * (0.28 + 0.78 * rProj.left);
              odorRight += plume * 0.14;
            } else {
              odorRight += plume * (0.28 + 0.78 * (-rProj.left));
              odorLeft += plume * 0.14;
            }
            if (rProj.fwd < -0.6 && Math.abs(rProj.left) < 0.25) odorLeft += plume * 0.55;

            const probeX = x + fx * 2.15;
            const probeZ = z + fz * 2.15;
            const clear = computeArenaWhiskerClearance(arena, probeX, probeZ);
            let wallAhead = 0;
            if (clear < 1.35) {
              wallAhead = Math.min(1.55, Math.max(0, (1.35 - clear) / 1.15));
              const esc = computeArenaWallEscapeTarget(arena, x, z);
              const eProj = project(esc.x, esc.z);
              if (eProj.left >= 0) odorLeft += 0.55;
              else odorRight += 0.55;
            }

            const out = brain.step(dt, {
              odorLeft,
              odorRight,
              wallAhead,
              rewardCue: prox * 0.6,
              exploreDrive: 0.62,
            }, { maxSpeed: 9.2 });

            yaw = Math.atan2(Math.sin(yaw + out.turnRate * dt), Math.cos(yaw + out.turnRate * dt));
            const track = 1 - Math.exp(-6 * dt);
            vx += (Math.sin(yaw) * out.forwardSpeed - vx) * track;
            vz += (Math.cos(yaw) * out.forwardSpeed - vz) * track;
            x += vx * dt;
            z += vz * dt;
            const res = resolveArenaPosition(
              arena, x, z, vx, vz, Math.sin(yaw), Math.cos(yaw),
              1.0, 1.0, 1.0, 0.0, 0.0, 9.2, wallFollowDir
            );
            x = res.x;
            z = res.z;
            vx = res.vx;
            vz = res.vz;
            wallFollowDir = res.wallFollowDir;
            minWallClearance = Math.min(minWallClearance, arenaSignedDistance(arena, x, z));
          }

          return {
            reachedReward,
            minDistToReward: Number(minDistToReward.toFixed(2)),
            minWallClearance: Number(minWallClearance.toFixed(2)),
          };
        }

        console.log(JSON.stringify({
          t_maze: runClosedLoop('t_maze', [0.0, 8.2], Math.PI),
          y_maze: runClosedLoop('y_maze', [-9.2, 5.3], Math.PI / 3),
          elevated_plus_maze: runClosedLoop('elevated_plus_maze', [-9.8, 0.0], Math.PI / 2),
          three_chamber: runClosedLoop('three_chamber', [2.2, 2.0], -Math.PI / 2),
          light_dark_box: runClosedLoop('light_dark_box', [6.2, 2.5], -Math.PI / 2),
        }));
        """
    )
    for arena_id, metrics in result.items():
        assert metrics["minWallClearance"] >= 1.1, (
            f"Wall breach in {arena_id}: {metrics}"
        )
        assert metrics["reachedReward"] is True, (
            f"Fly-Brain connectome did not reach reward in {arena_id}: {metrics}"
        )
