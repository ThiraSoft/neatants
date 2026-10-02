# neatants

Ant colonies that learn to survive through NEAT neuroevolution, in a small Ebiten world full of monsters, spells and a supermarket.

<!-- screenshot: docs/screenshot.png -->

## What it is

Several rival ant colonies share a map with a monster cave. Every ant is driven by a neural network. Nobody trains these networks by hand: ants are born, live, die, and the genomes of the ones that did well are bred into the next generation. You can watch it in a window, or run it headless on all your cores and come back later to see what evolved.

The simulation is continuous (steady-state): there are no discrete generations. A new ant is bred whenever a queen hatches an egg.

## How evolution works

**Genome.** Brains are NEAT genomes (package `neat/`): connection genes with global innovation numbers, structural mutations (add connection, add node, remove node, toggle connection), adaptive weight mutation, crossover aligned on innovation numbers, and a compatibility distance. On top of plain NEAT, genomes can grow memory cells and per-connection Hebbian plasticity (Oja's rule). Each genome also carries heritable traits: element affinities, size, speed, instinct, courage, curiosity, loyalty and diligence. New genomes start minimal by default (`ant_hidden: 0`), so the structure comes from mutation.

**What ants sense.** Six sectors around the ant, with six channels each (food, pheromone trail, alarm, and threat signals), plus inputs for the nest, body state, the clock, and vectors to the nearest food, threat and sisters.

**What ants do.** The network has 8 outputs: steering, speed, casting a spell, and a few more. In `brain_mode: full` the network drives everything. In `hybrid` mode an instinct layer picks a role (forage, return, fight, flee, guard, explore, rescue, raid, shop, watch the queen's live stream) and the network is blended with it. Spells come from the ant's element: Fire (fireball), Frost (slow), Storm (lightning), Earth (shield), Life (healing). Casting uses mana, and an element everyone uses runs thin, which lets minority elements survive. Casting into the void is penalised.

**Selection** (`selection` in `config.yml`):

- `colony`: half the fitness is what the colony earned while the ant lived (wealth per member), half is her own contribution. Default.
- `advanced`: individual contribution only: deliveries, monster kills, survival, plus learning rewards.
- `classic`: the older fixed weighted score.

Learning rewards (picking food up, progress toward home, fighting) fade over `shaping_fade` ticks down to a floor of `shaping_floor`, so early brains get guidance and late ones are judged on what keeps the colony alive.

**Breeding.** Each colony keeps a hall of fame and a pool of recent deaths. Children come from tournament selection (k=4) inside a species, with crossover, champions of large species copied unchanged, fitness sharing, and stagnant species retired. Species use NEAT compatibility distance, with a threshold that drifts to keep 3 to 8 species. Colonies are islands: rare migrants move between them, and the hall ages its fitness so a lucky outlier does not reign forever.

**Monsters evolve too.** With `monster_brains: true`, monsters share a NEAT gene pool that steers them on top of their script (turn, speed, chase ants or hit the nest). Their fitness is the harm they cause. A threat level adjusts wave size and strength so neither side crushes the other.

## Mechanics

- **Colonies.** A queen lays eggs, ants eat and deliver food, nests have hit points. A destroyed colony is refounded after a delay, from its own lineage, the best one, or a fresh one (`refound_from`).
- **Food.** Bushes spawn food, never close to nests. The empty zone around nests starts small and widens as colonies do well (`food_gap_adaptive`). Pheromone trails decay over time.
- **Monster waves.** Spiders, beetles, wasps, centipedes and golems come out of a cave in waves, with a boss every few waves. The cave can move after each wave.
- **Cave fever.** Old monsters can catch a contagious disease that drains max HP and spreads to neighbours.
- **Raids and rivalry.** Bold, disloyal ants of strong colonies can plunder rival nests.
- **Wallmart.** Colonies spend food on insecticide packs, bought by shopper ants at the Wallmart or at a rival's kiosk. Some ants shoplift and get chased by a security guard. Rich colonies can pay for a scooter delivery. Once a day there is a Black Friday sale with a brawl in the parking lot.
- **Marriage.** Two neighbouring colonies that left each other alone long enough can marry: peace, shared pheromones, common defence, a food dowry, possibly a half-blood queen. Enough wounds between spouses end in divorce.
- **Silly extras** (toggle in config): frog rain and confetti storms, queens streaming live from the nest, which buffs the ants watching.
- **Day and night cycle and weather.**

## Requirements

- Go 1.23 or newer (see `go.mod`)
- For the game window, Ebiten's system dependencies. On Linux you need a C toolchain, OpenGL, X11 and ALSA headers. See the [Ebiten install guide](https://ebitengine.org/en/documents/install.html) for your distribution.
- The headless runner does not open a window.

## Build and run

```sh
make build     # builds ./neatants and ./neatants-headless
make run       # builds and starts the game
go run ./cmd/neatants   # same without make
```

`make help` lists every target: `build`, `game`, `headless`, `test`, `test-long`, `run`, `evolve`, `clean`, `clean-saves`.

The game reads `config.yml` from the working directory (override with `-config path`), so run it from the repository root. Lineages are saved to `saves/colonies.json` (with a `.prev.json` backup) periodically and on exit, and loaded on the next start.

Game keys: Space pause, 1 to 5 speed, U turbo (simulate flat out, the window shows a summary), P pheromones, M simple view, C cinematic, H help, Tab hide HUD, F fullscreen, N call the next wave, Esc deselect. Arrow keys or WASD move the camera.

## Headless evolution

```sh
make evolve               # one world per CPU core, until Ctrl+C
make evolve WORLDS=1      # a single world
./neatants-headless -ticks 2000000
./neatants-headless -gpu  # 96 worlds, brains thinking on the GPU
```

Flags:

| Flag | Default | Meaning |
| --- | --- | --- |
| `-ticks` | 0 | ticks to simulate per world, 0 runs until Ctrl+C |
| `-worlds` | number of CPUs | worlds evolving in parallel. With 1, a single world spreads brain computation over cores |
| `-migrate` | 20000 | ticks between migrations of champions from one world to the next |
| `-report` | 10s | interval between progress reports |
| `-gpu` | off | think for every world's brains on the GPU through Vulkan, in batches. Falls back to the CPU without a usable device. Sets `-worlds` to 96 unless you give it |
| `-groups` | 3 | with `-gpu`, how many batches the worlds are split into, so the CPU steps some groups while the card thinks for another |
| `-config` | `config.yml` | configuration file |

With several worlds, every world starts from the same save and evolves on its own goroutine. Each colony's champion periodically sails to the next world, and the saved file merges the best genomes of all worlds. Ctrl+C saves and exits. The game then picks the lineages up from `saves/`.

### Running on the GPU

`-gpu` needs a Vulkan driver (Mesa RADV, or a vendor driver). There is nothing to compile: golem loads libvulkan at run time, without cgo.

It pays off with many worlds. On the author's machine (i7-9700K, RX 9070 XT, 5000 ticks), CPU only with 8 worlds runs at x87. With `-gpu` and 96 worlds it reaches about x330. Here xN is the sum over all worlds of simulated ticks per second, divided by 60.

Networks compute in float32 on both the CPU and the GPU, so a lineage evolved on the GPU behaves the same in the game, give or take float rounding.

## Configuration

`config.yml` is commented. Main sections:

- Colonies: number of colonies, starting ants, nest HP, ant speed, vision, lifespan, `think_every`, `brain_mode`, `selection`, shaping, instinct, refounding, hall of fame size.
- Food: bushes, spawn rate, `food_gap_adaptive`.
- Pheromones: decay.
- Monsters: wave timing and size, adaptive waves and threat signal, `monster_brains`, boss frequency, lifespan, cave fever.
- Wallmart: prices, pack sizes, delivery fee, Black Friday.
- Absurdities: weather, live-streaming queens.
- Marriages: delay, grief limit, cave moving.
- Misc: save interval, day length, rain.

## Project layout

```
cmd/neatants/            game entry point
cmd/neatants-headless/   headless evolution runner
neat/                    NEAT genomes, networks, crossover, speciation distance
internal/sim/            the simulation (no Ebiten)
internal/render/         Ebiten rendering, HUD, shaders
internal/headless/       multi-world headless runner
internal/gpubrain/       GPU brain evaluator: compute kernel, arena of network slots, batches
config.yml               default configuration
saves/                   saved lineages (git-ignored)
```

## Tests

```sh
make test         # go vet and the fast tests
make test-long    # slow tests too (sets NEATANTS_LONG_TESTS=1)
```

## License

MIT, see [LICENSE](LICENSE).
