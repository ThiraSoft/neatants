package sim

import (
	"fmt"
	"github.com/ThiraSoft/neatants/neat"
	"math"
	"math/rand"
)

type MonsterKind int

const (
	MonSpider MonsterKind = iota
	MonBeetle
	MonWasp
	MonCentipede
	MonGolem
	NumMonsterKinds
)

type monsterSpec struct {
	Name   string
	HP     float64
	Speed  float64
	Dmg    float64
	Radius float64
	CD     int
	Fly    bool
	Drop   int
}

var MonsterSpecs = [NumMonsterKinds]monsterSpec{
	MonSpider:    {"Spider", 50, 1.35, 1.6, 9, 40, false, 4},
	MonBeetle:    {"Armored Beetle", 140, 0.8, 3.2, 13, 60, false, 8},
	MonWasp:      {"Wasp", 36, 2.3, 1.2, 8, 32, true, 3},
	MonCentipede: {"Ancestral Centipede", 1000, 1.05, 5, 20, 45, false, 45},
	MonGolem:     {"Magma Golem", 520, 0.6, 8, 24, 80, false, 30},
}

type Monster struct {
	ID       int
	Kind     MonsterKind
	Pos      Vec2
	Prev     Vec2
	Angle    float64
	Heading  float64
	Gait     float64
	HP       float64
	MaxHP    float64
	Speed    float64
	Dmg      float64
	Radius   float64
	AttackCD int
	Target   int
	Fx       Effects
	Age      int
	MaxAge   int
	Enraged  bool
	Flash    float64
	Alive    bool
	Fly      bool
	Segments []Vec2
	Spawn    float64
	Lunge    float64

	Genome        *neat.Genome
	Net           *neat.Network
	Hits          float64 // ants struck and killed (see monKillReward), the brain's fitness
	LastHurtByAnt int
	sense         [MonInputs]float64
	out           [MonOutputs]float64
	feltHP        float64 // HP at the last thought, for the pain input
}

func (w *World) updateWaves() {
	alive := 0
	for _, m := range w.Monsters {
		if m.Alive {
			alive++
		}
	}
	if w.WaveAlive > 0 && alive == 0 && len(w.SpawnQueue) == 0 {
		w.log(fmt.Sprintf("Wave %d was repelled.", w.Wave), [3]float64{0.7, 1, 0.7})
	}
	w.WaveAlive = alive + len(w.SpawnQueue)

	if w.Tick >= w.NextWave {
		if Cfg.CaveMoves && w.Wave > 0 {
			w.moveCave()
		}
		w.Wave++
		w.NextWave = w.Tick + Cfg.WaveInterval
		w.adaptFoodGap()
		n := Cfg.WaveBase + int(float64(w.Wave-1)*Cfg.WaveGrowth)
		if Cfg.AdaptiveWaves {
			w.adaptThreat()
			n = max(Cfg.WaveMin, int(math.Round(float64(Cfg.WaveBase)*w.Threat)))
		}
		if Cfg.WavePerAnt > 0 {
			// The waves scale with the colonies: never more than a set
			// number of monsters per living ant, counting those still around.
			ants, alive := 0, 0
			for _, a := range w.Ants {
				if a.Alive {
					ants++
				}
			}
			for _, m := range w.Monsters {
				if m.Alive {
					alive++
				}
			}
			n = max(2, min(n, int(float64(ants)*Cfg.WavePerAnt)-alive))
		}
		w.SpawnQueue = w.SpawnQueue[:0]
		for i := range n {
			k := MonSpider
			r := rand.Float64()
			switch {
			case w.Wave >= 3 && r < 0.3:
				k = MonWasp
			case w.Wave >= 2 && r < 0.55:
				k = MonBeetle
			}
			_ = i
			w.SpawnQueue = append(w.SpawnQueue, k)
		}
		msg := fmt.Sprintf("Wave %d: %d creatures burst out of the cave.", w.Wave, n)
		if w.Wave%Cfg.BossEvery == 0 {
			w.SpawnQueue = append(w.SpawnQueue, MonCentipede)
			msg = fmt.Sprintf("Wave %d: the ancestral Centipede awakens!", w.Wave)
		} else if w.Wave >= 4 && w.Wave%Cfg.BossEvery == 3 {
			w.SpawnQueue = append(w.SpawnQueue, MonGolem)
			msg = fmt.Sprintf("Wave %d: a Magma Golem emerges from the depths.", w.Wave)
		}
		w.log(msg, [3]float64{1, 0.45, 0.35})
		w.emit(Event{Kind: EvWave, Pos: w.Cave, Val: float64(w.Wave)})
		// Target the colony nearest to the cave, with some spread.
		w.WaveTarget = -1
	}

	if len(w.SpawnQueue) > 0 {
		w.spawnCD--
		if w.spawnCD <= 0 {
			w.spawnCD = 18
			k := w.SpawnQueue[0]
			w.SpawnQueue = w.SpawnQueue[1:]
			w.spawnMonster(k)
		}
	}
}

func (w *World) spawnMonster(k MonsterKind) {
	s := MonsterSpecs[k]
	scale := Cfg.MonsterHPMult * (1 + Cfg.WaveHPGrowth*float64(w.Wave-1))
	dmg := s.Dmg * (1 + 0.05*float64(w.Wave-1))
	if Cfg.AdaptiveWaves {
		// The threat replaces the endless growth per wave: half of it goes
		// to the number of monsters, half to their strength.
		scale = Cfg.MonsterHPMult * math.Sqrt(w.Threat)
		dmg = s.Dmg * math.Sqrt(w.Threat)
	}
	w.nextID++
	p := w.Cave.Add(Polar(rand.Float64()*math.Pi*2, rand.Float64()*30))
	m := &Monster{
		ID: w.nextID, Kind: k, Pos: p, Prev: p,
		Angle: math.Pi*0.75 + rand.NormFloat64()*0.4,
		HP:    s.HP * scale, MaxHP: s.HP * scale,
		Speed: s.Speed * (0.9 + rand.Float64()*0.2), Dmg: dmg,
		Radius: s.Radius, Alive: true, Fly: s.Fly, Target: w.pickTarget(p),
		MaxAge: int(float64(Cfg.MonsterMaxAge) * (0.8 + 0.4*rand.Float64())),
	}
	if k >= MonCentipede {
		m.MaxAge *= 2
	}
	m.Heading = m.Angle
	if Cfg.MonsterBrains {
		m.Genome = w.newMonsterGenome()
		m.Net = m.Genome.BuildNetwork()
	}
	if k == MonCentipede {
		for i := range 22 {
			m.Segments = append(m.Segments, p.Sub(Polar(m.Angle, float64(i+1)*15)))
		}
	}
	w.Monsters = append(w.Monsters, m)
	w.emit(Event{Kind: EvMonsterSpawn, Pos: p, Val: float64(k)})
}

// pickTarget chooses a living colony, whatever its distance to the cave:
// half the time uniformly at random, half the time in proportion to its
// wealth (workers plus reserves in eggs' worth), so monsters smell the rich
// nests and the poor ones get time to recover.
func (w *World) pickTarget(p Vec2) int {
	var alive []int
	total := 0.0
	for _, c := range w.Colonies {
		if c.Alive {
			alive = append(alive, c.ID)
			total += colonyWealth(c)
		}
	}
	if len(alive) == 0 {
		return -1
	}
	if total > 0 && rand.Float64() < targetByWealth {
		r := rand.Float64() * total
		for _, id := range alive {
			if r -= colonyWealth(w.Colonies[id]); r <= 0 {
				return id
			}
		}
	}
	return alive[rand.Intn(len(alive))]
}

// targetByWealth is the share of monsters drawn to a nest by its wealth.
const targetByWealth = 0.5

func colonyWealth(c *Colony) float64 {
	return float64(c.Pop) + c.Food/Cfg.AntCost
}

func (w *World) updateMonsters() {
	for _, m := range w.Monsters {
		if !m.Alive {
			continue
		}
		m.Prev = m.Pos
		m.Spawn = math.Min(1, m.Spawn+0.02)
		m.Flash *= 0.85
		m.Lunge *= 0.85
		if m.AttackCD > 0 {
			m.AttackCD--
		}
		m.Fx.Tick()
		m.Age++
		if m.Age > m.MaxAge {
			w.monsterDiesOfAge(m)
			continue
		}
		if m.Age > m.MaxAge*85/100 {
			m.Fx[FxOld] = 1
		}
		if d := m.Fx.dotDamage(w.Tick+m.ID, 3); d > 0 {
			w.damageMonster(m, d, nil)
			if !m.Alive {
				continue
			}
		}
		if Cfg.MonsterDisease {
			w.disease(m)
			if !m.Alive {
				continue
			}
		}
		// Bosses fly into a rage when badly hurt.
		if m.Kind >= MonCentipede && m.HP < m.MaxHP*0.35 {
			m.Fx.Add(FxMight, 2)
			m.Fx.Add(FxHaste, 2)
			if !m.Enraged {
				m.Enraged = true
				w.log(MonsterSpecs[m.Kind].Name+" flies into a mad rage!", [3]float64{1, 0.3, 0.2})
			}
		}
		if m.Target < 0 || !w.Colonies[m.Target].Alive {
			m.Target = w.pickTarget(m.Pos)
		}

		// Choose a goal: nearby ant, else target nest. A brain may prefer
		// the nest even with ants around.
		var goal Vec2
		hasGoal := false
		var prey *Ant
		preyD := 140.0
		if m.Kind == MonGolem {
			preyD = 90
		}
		senseR := preyD
		w.forAntsNear(m.Pos, preyD, func(a *Ant, d float64) {
			if d < preyD {
				prey, preyD = a, d
			}
		})
		o := w.monsterThink(m, prey, preyD, senseR)
		chase := prey != nil && (o == nil || o[2] >= 0.5 || m.Target < 0)
		if chase {
			goal, hasGoal = prey.Pos, true
		} else if m.Target >= 0 {
			goal, hasGoal = w.Colonies[m.Target].Pos, true
		}
		if hasGoal {
			to := goal.Sub(m.Pos)
			want := math.Atan2(to.Y, to.X)
			turn := 0.06
			if m.Kind == MonCentipede {
				want += math.Sin(float64(w.Tick)*0.05) * 0.5
			}
			if m.Kind == MonWasp {
				turn = 0.09
				want += math.Sin(float64(w.Tick)*0.05+float64(m.ID)) * 0.6
			}
			if o != nil {
				want += (o[0]*2 - 1) * monSteer
			}
			m.Angle = wrapAngle(m.Angle + ClampF(angDiff(want, m.Angle), -turn, turn))
		}
		speed := m.Speed * m.Fx.SpeedMul()
		if o != nil {
			speed *= 0.5 + 0.6*o[1]
		}
		if chase && preyD < m.Radius+6 {
			speed *= 0.3
		}
		// Separation from other monsters
		sep := Vec2{}
		w.forMonstersNear(m.Pos, m.Radius+10, func(o *Monster, d float64) {
			if o != m {
				diff := m.Pos.Sub(o.Pos)
				if l := diff.Len(); l > 0.01 {
					sep = sep.Add(diff.Scale(1 / l))
				}
			}
		})
		m.Pos = m.Pos.Add(Polar(m.Angle, speed)).Add(sep.Scale(0.6))
		m.Pos = clampWorld(m.Pos)
		if !m.Fly {
			m.Pos = w.pushOutOfPonds(m.Pos)
		}
		m.Gait += speed * 0.4
		m.Heading += angDiff(m.Angle, m.Heading) * 0.2
		if m.Segments != nil {
			prev := m.Pos
			for i := range m.Segments {
				d := m.Segments[i].Sub(prev)
				if l := d.Len(); l > 15 {
					m.Segments[i] = prev.Add(d.Scale(15 / l))
				}
				prev = m.Segments[i]
			}
		}

		// Attacks
		if m.AttackCD == 0 && !m.Fx.Has(FxStun) {
			atNest := m.Target >= 0 && w.Colonies[m.Target].Pos.Sub(m.Pos).Len() < NestRadius+m.Radius
			if prey != nil && preyD < m.Radius+7 && (chase || !atNest) {
				m.AttackCD = MonsterSpecs[m.Kind].CD
				m.Lunge = 1
				dmg := m.Dmg * m.Fx.DamageDealt()
				switch m.Kind {
				case MonGolem:
					// Ground slam: area damage that stuns
					w.forAntsNear(m.Pos, 55, func(a *Ant, d float64) {
						w.monsterHit(m, a, dmg*0.6)
						a.Fx.Add(FxStun, 40)
					})
					w.emit(Event{Kind: EvQuake, Pos: m.Pos, Val: 55, Elem: -1})
				case MonSpider:
					w.monsterHit(m, prey, dmg)
					prey.Fx.Add(FxPoison, 240)
				case MonCentipede:
					w.monsterHit(m, prey, dmg)
					prey.Fx.Add(FxPoison, 360)
				case MonWasp:
					w.monsterHit(m, prey, dmg)
					prey.Fx.Add(FxWeak, 300)
				default:
					w.monsterHit(m, prey, dmg)
				}
				w.emit(Event{Kind: EvMonsterAttack, Pos: LerpV(m.Pos, prey.Pos, 0.6), Val: float64(m.Kind)})
			} else if atNest {
				c := w.Colonies[m.Target]
				m.AttackCD = MonsterSpecs[m.Kind].CD
				m.Lunge = 1
				c.HP -= m.Dmg * 2 * m.Fx.DamageDealt()
				c.monsterHitAt = w.Tick
				w.lose(c, math.Min(c.Food, 0.5))
				c.HitFlash = 1
				w.emit(Event{Kind: EvNestHit, Pos: c.Pos, Colony: c.ID})
				w.nestAttacked(c)
			}
		}
	}
}

func (w *World) damageMonster(m *Monster, dmg float64, by *Ant) {
	if !m.Alive {
		return
	}
	m.HP -= dmg
	m.Flash = 1
	if by != nil {
		by.Impact += dmg
		m.LastHurtByAnt = w.Tick
	}
	if m.HP <= 0 {
		m.Alive = false
		w.monsterDied(m)
		w.TotalKilled++
		if by != nil {
			by.MonsterKills++
			w.Colonies[by.Colony].MonsterKills++
			// A kill sends the nearby sisters into a frenzy.
			w.forAntsNear(m.Pos, 80, func(a *Ant, d float64) {
				if w.allied(a.Colony, by.Colony) {
					a.Fx.Add(FxMight, 300)
				}
			})
		}
		s := MonsterSpecs[m.Kind]
		w.scatterFood(m.Pos, s.Drop, 2)
		w.emit(Event{Kind: EvMonsterDeath, Pos: m.Pos, Val: float64(m.Kind)})
		if m.Kind == MonCentipede || m.Kind == MonGolem {
			name := "a colony"
			if by != nil {
				name = "colony " + w.Colonies[by.Colony].Name
			}
			w.log(s.Name+" struck down by "+name+" !", [3]float64{1, 0.85, 0.4})
			for i, sp := range m.Segments {
				if i%3 == 0 {
					w.emit(Event{Kind: EvMonsterDeath, Pos: sp, Val: float64(MonSpider)})
				}
			}
		}
	}
}

// monsterDiesOfAge makes an old monster crumble; its body still feeds the ants.
func (w *World) monsterDiesOfAge(m *Monster) {
	m.Alive = false
	w.monsterDied(m)
	w.scatterFood(m.Pos, MonsterSpecs[m.Kind].Drop/2, 2)
	w.emit(Event{Kind: EvMonsterOld, Pos: m.Pos, Val: float64(m.Kind)})
	if m.Kind >= MonCentipede {
		w.log(MonsterSpecs[m.Kind].Name+" dies, beaten by the years.", [3]float64{0.8, 0.75, 0.7})
	}
}

// disease runs the cave fever: it strikes ageing monsters, drains a share of
// their max HP (so it matters for bosses too) and spreads to their neighbours.
func (w *World) disease(m *Monster) {
	const duration = 1500
	if !m.Fx.Has(FxPlague) {
		life := float64(m.Age) / float64(max(1, m.MaxAge))
		onset := Cfg.DiseaseOnset
		if life > onset {
			x := (life - onset) / (1 - onset)
			if rand.Float64() < 0.0015*x*x {
				m.Fx.Add(FxPlague, duration)
				if w.Tick-w.lastPlagueLog > 3000 {
					w.lastPlagueLog = w.Tick
					w.log("Cave fever gnaws at the oldest monsters.", Effect[FxPlague].Color)
				}
			}
		}
		return
	}
	phase := w.Tick + m.ID
	if phase%60 == 0 {
		w.damageMonster(m, m.MaxHP*Cfg.DiseaseDamage, nil)
		if !m.Alive {
			return
		}
	}
	if phase%30 == 0 {
		w.forMonstersNear(m.Pos, 50, func(o *Monster, d float64) {
			if o != m && !o.Fx.Has(FxPlague) && rand.Float64() < Cfg.DiseaseContagion {
				o.Fx.Add(FxPlague, duration)
				w.emit(Event{Kind: EvContagion, Pos: o.Pos, Pts: []Vec2{m.Pos}})
			}
		})
	}
}
