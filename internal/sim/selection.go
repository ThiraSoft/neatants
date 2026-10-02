package sim

import (
	"github.com/ThiraSoft/neatants/neat"
	"math"
	"math/rand"
)

// Advanced selection, following the usual neuroevolution guidelines, adapted
// to a steady-state world where ants are born and die continuously:
//
//   - fitness measures talent rather than luck: achievements per unit of life
//     (an ant eaten early is not punished for having had less time), plus a
//     dense shaping reward for picking food up;
//   - NEAT speciation (species.go): fitness sharing, children shared out
//     between species by their mean fitness, stagnant species retired,
//     champions of big species copied unchanged, crossover within a species;
//   - tournament selection (k=4) inside the chosen species;
//   - the hall of fame keeps at most a few genomes per species;
//   - fitness ageing in the hall so a lucky outlier cannot reign forever;
//   - island model: colonies evolve apart, with rare migrants between them;
//   - self-adaptive mutation strength (in the neat package).

const (
	tournamentK   = 4
	speciesDist   = 2.0 // NEAT compatibility below which two genomes share a species
	SpeciesCap    = 5   // hall members allowed per species
	hallAgeing    = 0.997
	eliteRate     = 0.10
	migrationRate = 0.02
	crossoverRate = 0.75
	// Lifetimes shorter than this count as this long. It matches the rate
	// horizon: an ant that starves young is judged on what she did over the
	// whole horizon, so going home to eat pays. With a shorter floor, a
	// forager dying at the first empty stomach scored like a long-lived one.
	minEvalTicks   = rateHorizon
	rateHorizon    = 20000.0
	survivalWeight = 1.0 / 12000
	groupScale     = 4.0 // brings the team term to the scale of the individual one
)

func Advanced() bool { return Cfg.Selection != "classic" }

// earn and lose move food in or out of a colony and keep its wealth ledger,
// used by the colony-level fitness.
func (w *World) earn(c *Colony, food float64) {
	c.Food += food
	c.Wealth += food
}

func (w *World) lose(c *Colony, food float64) {
	c.Food = math.Max(0, c.Food-food)
	c.Wealth -= food
}

// antFitness scores a dead ant.
//
// "colony" (default): an ant is worth what her colony produced while she was
// alive (wealth gained per member, per unit of time), plus how long she
// survived. No weight is put on any activity: whatever makes the colony
// prosper, foraging, defending or plundering, is what gets selected.
func (w *World) antFitness(a *Ant) float64 {
	if Cfg.Selection == "colony" {
		c := w.Colonies[a.Colony]
		gained := 0.0
		if c.Founded == a.BornFounded { // same colony, not a refounded one
			gained = c.Wealth - a.BornWealth
		}
		pop := math.Max(1, float64(a.BornPop+c.Pop)/2)
		group := gained / pop / math.Max(float64(a.Age), minEvalTicks) * rateHorizon
		// Half team reward, half individual contribution: a pure team reward
		// breeds free riders (only survival separates sisters, so cowards win).
		return 0.5*math.Max(0, group)*groupScale + 0.5*individualFitness(a, w.Shaping())
	}
	if !Advanced() {
		return float64(a.Delivered)*3 + float64(a.Kills)*2 + float64(a.MonsterKills)*5 + a.Impact/10 + float64(a.Age)/6000
	}
	return individualFitness(a, w.Shaping())
}

// Shaping is the weight of the learning rewards, from 1 for fresh lineages
// down to shaping_floor after shaping_fade ticks of evolution. They teach
// newborn brains to walk, forage and fight, then mostly step aside so that
// what really keeps a colony alive is selected. They never vanish: with
// deliveries alone the reward is too sparse, nothing leads an ant to food any
// more, and the lineages forget how to forage.
func (w *World) Shaping() float64 {
	if Cfg.ShapingFade <= 0 {
		return 1
	}
	return math.Max(Cfg.ShapingFloor, 1-float64(w.Evolved)/float64(Cfg.ShapingFade))
}

// individualFitness: achievements per unit of life, plus survival.
func individualFitness(a *Ant, shaping float64) float64 {
	// Food brought home and monsters slain always count, alike: a kill was
	// worth a tenth of a delivery once the shaping had faded, so fleeing
	// always won. The rest only guides learning and fades out: picking food
	// up, carrying it towards home (a signed reward: stepping back costs as
	// much as stepping forward, so it cannot be farmed), fighting rival ants,
	// damage and healing, and a penalty for spells cast into the void.
	learning := float64(a.Pickups) + a.HomeProgress/150 +
		float64(a.Kills) + a.Impact/15 -
		float64(a.Wasted)*wastePenalty
	achieved := float64(a.Delivered+a.MonsterKills)*4 + shaping*learning
	rate := achieved / math.Max(float64(a.Age), minEvalTicks) * rateHorizon
	return rate + float64(a.Age)*survivalWeight
}

// tournament returns the fittest of k genomes drawn at random from pool.
func tournament(pool []*neat.Genome, k int) *neat.Genome {
	best := pool[rand.Intn(len(pool))]
	for range k - 1 {
		if g := pool[rand.Intn(len(pool))]; g.Fitness > best.Fitness {
			best = g
		}
	}
	return best
}

// breedAdvanced makes a child genome for colony c.
func (w *World) breedAdvanced(c *Colony, pool []*neat.Genome) *neat.Genome {
	if rand.Float64() < migrationRate {
		var islands []*Colony
		for _, o := range w.Colonies {
			if o.Alive && o.ID != c.ID && len(o.Hall) > 0 {
				islands = append(islands, o)
			}
		}
		if len(islands) > 0 {
			child := tournament(islands[rand.Intn(len(islands))].Hall, tournamentK).Copy()
			child.Mutate()
			return child
		}
	}
	return breedSpeciated(&c.Species, pool, w.Evolved)
}

// mergeEval folds the life of a cloned champion into its hall record: the
// fitness becomes the mean over its lives and the hall is sorted again. It
// reports whether g belonged to a genome already there.
func mergeEval(hall []*neat.Genome, g *neat.Genome) bool {
	id := g.Lineage()
	for i, h := range hall {
		if h.Lineage() != id {
			continue
		}
		n := float64(max(1, h.Evals))
		h.Fitness = (h.Fitness*n + g.Fitness) / (n + 1)
		h.Evals = int(n) + 1
		for ; i > 0 && hall[i-1].Fitness < h.Fitness; i-- {
			hall[i-1], hall[i] = hall[i], hall[i-1]
		}
		for ; i < len(hall)-1 && hall[i+1].Fitness > h.Fitness; i++ {
			hall[i+1], hall[i] = hall[i], hall[i+1]
		}
		return true
	}
	return false
}

// insertHallSpeciated keeps the hall sorted by fitness, ages old records and
// caps each species (genomes closer than dist) so the hall stays diverse.
func InsertHallSpeciated(hall *[]*neat.Genome, g *neat.Genome, dist float64) {
	for _, h := range *hall {
		h.Fitness *= hallAgeing
	}
	same, worst := 0, -1
	for i, h := range *hall {
		if neat.Compatibility(g, h) < dist {
			same++
			if worst < 0 || h.Fitness < (*hall)[worst].Fitness {
				worst = i
			}
		}
	}
	if same >= SpeciesCap {
		if g.Fitness <= (*hall)[worst].Fitness {
			return // its species is already well represented by better genomes
		}
		*hall = append((*hall)[:worst], (*hall)[worst+1:]...)
	}
	InsertHallOfFame(hall, g)
}
