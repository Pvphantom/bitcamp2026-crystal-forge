# Crystal Forge Minecraft Bridge

This directory is now the start of the Fabric mod plus the handoff notes
between the FastAPI backend and Minecraft. The first implementation target is a
control-room scene in a superflat Java Edition world.

## Source of truth

The Fabric side should consume:

- `POST /api/minecraft/workflow`

This route wraps the unified workflow backend and returns the dedicated
`minecraft_v1` schema.

## Intended scene layout

- Center: lattice platform
- Left wall: parameter controls
- Right wall: CorrMap and workflow decision
- Back wall: global observables
- Rear chamber: VQE activation zone
- Overhead ring: QProbe and Adaptive QProbe nodes

## Current backend contract

Top-level sections in `minecraft_v1`:

- `scene`
- `problem`
- `routing`
- `workflow`
- `regime`
- `observables`
- `trust`
- `solvers`
- `measurement`
- `visualization_hints`

The contract is model-family agnostic across:

- `hubbard`
- `tfim`

## Rendering expectations

### Hubbard

- site rendering: pedestal style
- primary site value: `Sz`
- secondary site value: `double_occ`
- occupancy label provided for discrete color mapping
- bond kind: `spin_correlation`

### TFIM

- site rendering: pillar style
- primary site value: `Mz`
- secondary site value: `Mx`
- bond kind: `ZZ`

## Important workflow semantics

- `workflow.active_path_type = cheap`
  - no escalation
  - QProbe hidden

- `workflow.active_path_type = quantum`
  - escalated to VQE
  - show VQE chamber
  - show QProbe ring

- `workflow.active_path_type = exact_fallback`
  - no quantum route available locally
  - show exact fallback state
  - QProbe may be hidden or marked benchmark-only depending on design

## What exists now

The scaffold currently includes:

- Gradle/Fabric project files
- mod id: `crystalforge`
- a backend bridge using Java's `HttpClient`
- a payload store for the most recent backend response
- a render placeholder hook
- server commands:
  - `/crystalforge refresh`
  - `/crystalforge preset tfim_safe`
  - `/crystalforge preset tfim_quantum`
  - `/crystalforge preset hubbard_fallback`
  - `/crystalforge status`

Current behavior:

1. Fetches `POST /api/minecraft/workflow`
2. Parses the JSON payload
3. Stores the most recent payload in memory
4. Prints a short summary in chat
5. Builds a first-pass control-room scene directly in-world:
   - room shell
   - center lattice platform
   - Hubbard pedestals or TFIM pillars
   - bond bridges
   - right-side trust indicator
   - rear path indicator

## Suggested next Fabric milestones

1. Expand the simple workflow indicator into the full right-side workflow wall
2. Add the full back observables wall
3. Add the VQE chamber and overhead QProbe ring
4. Add button-based parameter controls that resubmit the route
5. Add animation instead of hard overwrites

## Example request

```json
{
  "model_family": "tfim",
  "Lx": 2,
  "Ly": 2,
  "parameters": { "J": 1.0, "h": 0.8, "g": 0.0 },
  "qprobe_targets": ["Mz", "ZZ_nn", "Mstag2"],
  "qprobe_tolerance": 0.03,
  "qprobe_shots_per_group": 4000,
  "qprobe_readout_flip_prob": 0.02,
  "qprobe_seed": 7
}
```

## How to test this with your Minecraft account

Nothing here is tied to a specific Minecraft account. Any Java Edition account
can use the same mod jar or dev client as long as:

- the game version matches `1.20.4`
- Fabric Loader is used
- the backend is running locally

### Local test flow

1. Start the FastAPI backend from the repo root.
2. Make sure the backend is reachable at:
   - `http://127.0.0.1:8000/api/minecraft/workflow`
3. Launch Minecraft Java Edition with Fabric for `1.20.4`.
4. Enter or create a superflat world.
5. Run:
   - `/crystalforge refresh`
   - or `/crystalforge preset tfim_quantum`
   - or `/crystalforge preset tfim_safe`
   - or `/crystalforge preset hubbard_fallback`
6. The scene should render near:
   - `(0, 64, 0)`

### If another person uses the GitHub repo

They need:

- a Java Edition Minecraft account
- Java 17
- Minecraft `1.20.4`
- Fabric Loader
- this repo cloned locally
- the backend running on their machine

They do **not** need your Minecraft account.

### Backend URL override

By default the mod uses:

- `http://127.0.0.1:8000/api/minecraft/workflow`

If someone wants a different backend URL, they can override it with:

- JVM property:
  - `-Dcrystalforge.backendUrl=http://HOST:PORT/api/minecraft/workflow`
- or environment variable:
  - `CRYSTALFORGE_BACKEND_URL=http://HOST:PORT/api/minecraft/workflow`

## Build notes

This scaffold does not yet include a Gradle wrapper, and this machine does not
currently have `gradle` installed. That means I set up the Fabric project
structure and source files, but I did not run a local mod build here.

To build or run it locally, either:

- install Gradle and run it in `minecraft/`
- or generate a Gradle wrapper in `minecraft/`
- or import the module into IntelliJ with Fabric Loom support
