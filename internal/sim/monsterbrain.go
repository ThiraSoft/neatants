package sim

import (
	"github.com/ThiraSoft/neatants/neat"
	"math"
)

// Monsters evolve too. The cave keeps one gene pool shared by every kind of
// monster; a brain does not replace the script but steers on top of it: it
// bends the scripted heading, doses the speed and chooses between chasing an
// ant and battering the nest. Its fitness is the harm it did (hits on ants
// and on nests), so living longer and choosing better targets both pay.
//
// Coevolution only works while neither side crushes the other, or the losing
// side gets no gradient at all. The threat level keeps the balance: each wave
// it grows while monsters cause less than threat_deaths of the ants' deaths,
// and shrinks when they cause more or when nests fall. (threat_signal: kills
// is the former rule, aiming at a share of monsters slain: ants that fled
// let the monsters die of age, which read as "too strong" and weakened them
// down to the floor.)

const (
	MonInputs  = 19
	MonOutputs = 3 // turn, speed, focus (chase ants rather than the nest)
	monHidden  = 0

	monSteer  = math.Pi / 2 // how far a brain may bend the scripted heading
	threatMin = 0.15
	threatMax = 6.0
	threatK   = 0.6  // reaction of the threat to the kill share, per wave
	nestLossK = 0.85 // threat multiplier for each nest lost since last wave
	// A nest razed by monsters weighs as much as this many ants killed by
	// them in the threat signal. Monsters that batter nests rather than ants
	// otherwise looked harmless, so the threat kept rising while they razed
	// a nest every few ant lifetimes, starving and toppling the colonies.
	nestLossDeaths = 8.0
	slainMemo      = 300 // ticks during which an ant's hit claims a monster's death
	threatSmooth   = 0.8 // memory of the death shares from one wave to the next
)

// newMonsterGenome breeds a brain from the cave's pool.
func (w *World) newMonsterGenome() *neat.Genome {
	pool := append(append([]*neat.Genome{}, w.MonHall...), w.MonRecent...)
	var g *neat.Genome
	if len(pool) >= 2 {
		g = breedSpeciated(&w.MonSpecies, pool, w.Evolved)
	} else {
		g = neat.NewGenomeWithHidden(0, MonInputs, MonOutputs, monHidden)
		g.Mutate()
	}
	w.nextID++
	g.ID = w.nextID
	g.Fitness = 0
	return g
}

// monsterDied records the fitness of a dead monster's brain and tells the
// threat level who won.
func (w *World) monsterDied(m *Monster) {
	if m.LastHurtByAnt > 0 && w.Tick-m.LastHurtByAnt < slainMemo {
		w.slain++
	} else {
		w.outlived++
	}
	if m.Genome == nil || m.Hits <= 0 {
		return
	}
	g := m.Genome.Copy()
	g.ID, g.Fitness, g.Evals = m.Genome.ID, m.Hits, 1
	if mergeEval(w.MonHall, g) {
		return
	}
	w.MonRecent = append(w.MonRecent, g)
	if len(w.MonRecent) > Cfg.SurvivorPool {
		w.MonRecent = w.MonRecent[1:]
	}
	InsertHallSpeciated(&w.MonHall, g, w.MonSpecies.Threshold())
}

// adaptThreat runs at each new wave.
func (w *World) adaptThreat() {
	if w.Threat <= 0 {
		w.Threat = 1
	}
	err := 0.0
	if Cfg.ThreatSignal == "kills" {
		if total := w.slain + w.outlived; total >= 3 {
			err += float64(w.slain)/float64(total) - Cfg.WaveBalance
		}
		if w.antDeaths >= 5 {
			err -= math.Max(0, float64(w.antsByMonsters)/float64(w.antDeaths)-0.5)
		}
	} else {
		// Aim at a share of the ants' deaths. Ants that flee no longer make
		// the monsters weaker: harmless monsters get stronger until the
		// colonies have to defend themselves. The share is smoothed over a
		// few waves: one massacre must not undo ten waves of slow growth.
		razed := nestLossDeaths * float64(w.nestsLost)
		w.threatDeaths = threatSmooth*w.threatDeaths + float64(w.antDeaths) + razed
		w.threatByMon = threatSmooth*w.threatByMon + float64(w.antsByMonsters) + razed
		if w.threatDeaths >= 5 {
			err = Cfg.ThreatDeaths - w.threatByMon/w.threatDeaths
		}
	}
	if Cfg.ThreatSignal != "kills" {
		// Monsters dying of age were never confronted: one more reason to
		// send stronger ones.
		if total := w.slain + w.outlived; total >= 3 {
			err += Cfg.ThreatAge * float64(w.outlived) / float64(total)
		}
	}
	w.Threat *= math.Exp(threatK * err)
	w.Threat *= math.Pow(nestLossK, float64(w.nestsLost))
	w.Threat = ClampF(w.Threat, threatMin, threatMax)
	w.slain, w.outlived, w.nestsLost = 0, 0, 0
	w.antDeaths, w.antsByMonsters = 0, 0
}

// monsterThink fills the brain's senses and returns its outputs, or nil
// when monsters have no brain.
func (w *World) monsterThink(m *Monster, prey *Ant, preyD, senseR float64) []float64 {
	if m.Net == nil {
		return nil
	}
	in := m.sense[:]
	rel := func(p Vec2) (float64, float64) {
		d := p.Sub(m.Pos)
		a := angDiff(math.Atan2(d.Y, d.X), m.Angle)
		return math.Sin(a), math.Cos(a)
	}
	clear(in)
	in[0] = m.HP / m.MaxHP
	if m.AttackCD == 0 {
		in[1] = 1
	}
	in[5] = 1
	if prey != nil {
		in[2] = 1
		in[3], in[4] = rel(prey.Pos)
		in[5] = preyD / senseR
	}
	ants := 0
	w.forAntsNear(m.Pos, senseR, func(a *Ant, d float64) { ants++ })
	in[6] = math.Min(1, float64(ants)/10)
	in[9] = 1
	if m.Target >= 0 {
		c := w.Colonies[m.Target]
		in[7], in[8] = rel(c.Pos)
		in[9] = math.Min(1, c.Pos.Sub(m.Pos).Len()/1500)
		in[15] = c.HP / Cfg.NestMaxHP
	}
	in[12] = 1
	best := 200.0
	w.forMonstersNear(m.Pos, best, func(o *Monster, d float64) {
		if o != m && d < best {
			best = d
			in[10], in[11] = rel(o.Pos)
			in[12] = d / 200
		}
	})
	in[13] = m.Radius / 24
	if m.Fly {
		in[14] = 1
	}
	in[16] = w.Clock
	in[17] = float64(m.Age) / float64(max(1, m.MaxAge))
	in[18] = pain(m.feltHP, m.HP, m.MaxHP)
	m.feltHP = m.HP
	return m.Net.Activate(in)
}
