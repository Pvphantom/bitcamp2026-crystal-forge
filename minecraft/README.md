# Crystal Forge — Minecraft Mod

Fabric 1.20.4 mod (Java 17) that turns the backend's lattice / routing /
observable payload into a **walkable 3D control room** inside a superflat world.
Parameter changes made by clicking blocks re-hit the backend and live-update
the scene — no `/command` required.

- Minecraft **1.20.4** (Java Edition)
- Fabric Loader **0.15.11**, Fabric API **0.97.0+1.20.4**
- Yarn mappings **1.20.4+build.3**
- Java **17**
- Gradle wrapper + Fabric Loom included

---

## Quick start

```bash
cd minecraft
./gradlew runClient          # launches Minecraft with the mod loaded
```

Then inside the game:

1. Create / enter a superflat world.
2. On first join, the mod auto-fetches the default TFIM preset and builds the
   scene near `(0, 64, 0)`. You'll see a chat message: *"Crystal Forge: scene
   ready. Right-click the control wall to interact."*
3. Walk to the control wall (left side of the lattice platform) and right-click
   the colored blocks — the scene re-renders live.

If auto-init fails ("backend not running"), start the backend (see repo root
`README.md`) and run `/crystalforge refresh` in chat.

---

## Scene layout

The backend payload dictates exact geometry. Default cardinal layout, with the
platform centered at `scene.origin`:

| Location          | Purpose                                              |
|-------------------|------------------------------------------------------|
| Center platform   | Lattice — pedestals (Hubbard) or pillars (TFIM)      |
| Left wall         | **Clickable control wall** — model toggle + ± params |
| Right wall        | CorrMap verdict + workflow path                      |
| Back wall         | Global observables as colored bars                   |
| Rear chamber      | VQE activation zone (only on `quantum` path)         |
| Overhead ring     | QProbe / Adaptive QProbe nodes                       |

### Control wall bindings

All clicks are resolved **relative to `scene.origin`**, so the wall works
regardless of where the room was generated.

| Block color / row                    | Action                                     |
|--------------------------------------|--------------------------------------------|
| Blue (top)                           | Switch to TFIM defaults                    |
| Purple (top)                         | Switch to Hubbard defaults                 |
| Lime (bottom)                        | Re-run current request                     |
| Orange (three parameter rows, lower) | Decrement that parameter by `stepSize`     |
| Green  (three parameter rows, upper) | Increment that parameter by `stepSize`     |

Parameter rows:

- **TFIM**: `J`, `h`, `g`
- **Hubbard**: `t`, `U`, `mu`

---

## Chat commands

```
/crystalforge refresh              # re-run current request
/crystalforge preset <name>        # switch preset by name
/crystalforge status               # re-render last payload without fetching
```

Known preset names: `tfim_safe`, `tfim_quantum`, `hubbard_fallback` — see
`CrystalForgeRequests.java` for the canonical list.

---

## How it talks to the backend

The mod only knows one endpoint:

```
POST <backendUrl>
```

Default:

```
http://127.0.0.1:8000/api/minecraft/workflow
```

Override via JVM property or env var:

```
-Dcrystalforge.backendUrl=http://HOST:PORT/api/minecraft/workflow
CRYSTALFORGE_BACKEND_URL=http://HOST:PORT/api/minecraft/workflow
```

HTTP is done on a **daemon worker thread** (`cf-refresh`, `cf-init`,
`cf-cmd-*`) — the server tick is never blocked. Once the response arrives,
the render step is scheduled back onto the server thread via
`server.execute(...)`.

### Payload contract (`minecraft_v1`)

Top-level sections:

- `scene` — `origin { x, y, z }`, per-site blocks, bonds, walls
- `problem` — model family, lattice size, parameters
- `routing` — CorrMap verdict + confidence
- `workflow` — active path: `cheap` | `quantum` | `exact_fallback`
- `regime`, `observables`, `trust`, `solvers`, `measurement`,
  `visualization_hints`

Path semantics:

| `workflow.active_path_type` | Scene                                                |
|-----------------------------|------------------------------------------------------|
| `cheap`                     | No VQE chamber, QProbe ring hidden                   |
| `quantum`                   | VQE chamber lit, QProbe ring shown                   |
| `exact_fallback`            | Exact-fallback indicator, QProbe marked benchmark    |

Site rendering:

- **Hubbard** — pedestals, primary `Sz`, secondary `double_occ`, bonds
  colored by `spin_correlation`.
- **TFIM** — pillars, primary `Mz`, secondary `Mx`, bonds colored by `ZZ`.

---

## Source layout

```
minecraft/
├── build.gradle            Loom + Fabric config, Java 17 pinned
├── gradle.properties       MC / Loader / Yarn / Fabric API versions
├── gradlew, gradlew.bat    Gradle wrapper
└── src/main/java/io/crystalforge/minecraft/
    ├── CrystalForgeMod.java           ModInitializer — commands + auto-init
    ├── CrystalForgeClientMod.java     Client-side entrypoint
    ├── CrystalForgeControls.java      UseBlockCallback → click → refresh
    ├── CrystalForgeBridge.java        HTTP client + render dispatcher
    ├── CrystalForgeRenderer.java      Builds the in-world scene from payload
    ├── CrystalForgePayloadStore.java  Caches latest payload + scene origin
    ├── CrystalForgeSession.java       Current request + parameter state
    ├── CrystalForgeRequests.java      Named preset factories
    ├── CrystalForgeWorkflowRequest.java  Request wrapper
    └── CrystalForgeConfig.java        Backend URL resolution
```

---

## Multiplayer / sharing

Nothing is tied to a specific Minecraft account. Anyone can run this mod if
they have:

- A Java Edition Minecraft account
- Java 17
- Minecraft 1.20.4 + Fabric Loader
- This repo cloned locally
- The backend running (locally or reachable via `CRYSTALFORGE_BACKEND_URL`)

---

## Build notes

```bash
./gradlew build              # produces build/libs/crystal-forge-minecraft-<ver>.jar
./gradlew runClient          # dev client with mod loaded
./gradlew runServer          # dev server (if you want to test multiplayer)
```

Or import the `minecraft/` directory into IntelliJ with the Fabric Loom
plugin and use the provided run configs.

---

## Roadmap / next milestones

1. Replace block-only indicators with signs, text displays, or richer
   `Display` entity panels.
2. Animate parameter transitions instead of hard overwrites.
3. Make the QProbe ring reflect measurement-group structure visually.
4. Add adaptive-step playback and solver-transition lighting.
5. Add target / tolerance controls on the left wall.
