package sim

import (
	"math"
	"math/rand"
)

type Fireball struct {
	Pos, Vel Vec2
	Colony   int
	Owner    *Ant
	Dmg      float64
	Life     int
	Alive    bool
	Trail    [12]Vec2
	Aimed    bool // shot by a network in full-brain mode (for hit statistics)
}

const (
	spellCost = 0.55
	castCD    = 100

	// A spell that touches nothing useful costs the caster: a longer
	// cooldown and some energy (felt through survival, hence also by the
	// colony-level fitness), plus a direct fitness penalty. Without it,
	// casting into the void is free once mana is full, and spam wins.
	wasteCD      = 180
	wasteEnergy  = 0.03
	wastePenalty = 0.2 // a fireball hitting one monster earns about 0.45
)

// castSpell fires the ant's elemental power if a meaningful target exists.
func (w *World) castSpell(a *Ant) {
	cast, landed := false, false
	switch a.Element {
	case ElemFire:
		cast, landed = w.castFire(a)
	case ElemFrost:
		cast, landed = w.castFrost(a)
	case ElemStorm:
		cast, landed = w.castStorm(a)
	case ElemEarth:
		cast, landed = w.castEarth(a)
	case ElemLife:
		cast, landed = w.castLife(a)
	}
	if cast {
		w.spendElement(a.Element)
		a.Mana -= spellCost
		a.CastCD = castCD
		a.LastCast = w.Tick
		a.Casts++
		if !landed {
			w.wasteCast(a)
		}
	} else {
		a.CastCD = 10
	}
}

// wasteCast charges the caster for a spell that touched nothing useful.
func (w *World) wasteCast(a *Ant) {
	a.Wasted++
	a.Energy -= wasteEnergy
	a.CastCD = max(a.CastCD, wasteCD-(w.Tick-a.LastCast))
}

// nearestFoe returns the closest monster or rival ant position within r.
func (w *World) nearestFoe(a *Ant, r float64) (Vec2, Vec2, bool) {
	best := r
	var pos, vel Vec2
	found := false
	w.forMonstersNear(a.Pos, r, func(m *Monster, d float64) {
		if d < best {
			best, pos, vel, found = d, m.Pos, m.Pos.Sub(m.Prev), true
		}
	})
	w.forAntsNear(a.Pos, r, func(b *Ant, d float64) {
		if w.hostile(a.Colony, b.Colony) && d < best*0.8 {
			best, pos, vel, found = d, b.Pos, b.Pos.Sub(b.Prev), true
		}
	})
	return pos, vel, found
}

// foeAhead returns the nearest foe within r that lies inside the aim-assist
// cone around the ant's heading (full-brain mode: the network aims).
func (w *World) foeAhead(a *Ant, r float64) (Vec2, Vec2, bool) {
	cone := Cfg.AimAssist * math.Pi / 180
	inCone := func(p Vec2) bool {
		d := p.Sub(a.Pos)
		return math.Abs(angDiff(math.Atan2(d.Y, d.X), a.Angle)) <= cone
	}
	best := r
	var pos, vel Vec2
	found := false
	w.forMonstersNear(a.Pos, r, func(m *Monster, d float64) {
		if d < best && inCone(m.Pos) {
			best, pos, vel, found = d, m.Pos, m.Pos.Sub(m.Prev), true
		}
	})
	w.forAntsNear(a.Pos, r, func(b *Ant, d float64) {
		if d < best && w.hostile(a.Colony, b.Colony) && inCone(b.Pos) {
			best, pos, vel, found = d, b.Pos, b.Pos.Sub(b.Prev), true
		}
	})
	return pos, vel, found
}

// aheadOf reports whether p lies inside the ant's aim-assist cone.
func (w *World) aheadOf(a *Ant, p Vec2) bool {
	d := p.Sub(a.Pos)
	return math.Abs(angDiff(math.Atan2(d.Y, d.X), a.Angle)) <= Cfg.AimAssist*math.Pi/180
}

// castFire reports landed = true: a fireball is judged when it explodes.
func (w *World) castFire(a *Ant) (bool, bool) {
	const speed = 5.5
	full := FullBrain()
	var dir float64
	if full {
		// The network aims: the fireball flies where the ant faces, nudged
		// onto a foe only if one sits inside the aim-assist cone.
		dir = a.Angle
		if p, v, ok := w.foeAhead(a, 190); ok {
			t := p.Sub(a.Pos).Len() / speed
			aim := p.Add(v.Scale(t)).Sub(a.Pos)
			dir = math.Atan2(aim.Y, aim.X)
		}
		w.AimShots++
	} else {
		p, v, ok := w.nearestFoe(a, 190)
		if !ok {
			return false, false
		}
		t := p.Sub(a.Pos).Len() / speed
		aim := p.Add(v.Scale(t)).Sub(a.Pos)
		dir = math.Atan2(aim.Y, aim.X)
	}
	b := &Fireball{
		Pos: a.Pos, Vel: Polar(dir, speed), Colony: a.Colony, Owner: a,
		Dmg: 9 * a.Power * a.Fx.DamageDealt(), Life: 70, Alive: true, Aimed: full,
	}
	for i := range b.Trail {
		b.Trail[i] = a.Pos
	}
	w.Balls = append(w.Balls, b)
	if !full {
		a.Angle, a.Heading = dir, dir // hybrid: the script turns the ant to shoot
	}
	w.emit(Event{Kind: EvFireCast, Pos: a.Pos, Colony: a.Colony, Val: dir})
	return true, true
}

func (w *World) updateFireballs() {
	live := w.Balls[:0]
	for _, b := range w.Balls {
		copy(b.Trail[1:], b.Trail[:len(b.Trail)-1])
		b.Trail[0] = b.Pos
		b.Pos = b.Pos.Add(b.Vel)
		b.Vel = b.Vel.Scale(1.004)
		b.Life--
		hit := b.Life <= 0 || w.InPond(b.Pos, -8)
		if !hit {
			w.forMonstersNear(b.Pos, 4, func(m *Monster, d float64) { hit = true })
		}
		if !hit {
			w.forAntsNear(b.Pos, 8, func(o *Ant, d float64) {
				if w.hostile(b.Colony, o.Colony) {
					hit = true
				}
			})
		}
		if hit {
			w.explode(b)
			continue
		}
		live = append(live, b)
	}
	clear(w.Balls[len(live):])
	w.Balls = live
}

func (w *World) explode(b *Fireball) {
	const r = 46
	owner := b.Owner
	if owner != nil && !owner.Alive {
		owner = nil
	}
	hit := false
	w.forMonstersNear(b.Pos, r, func(m *Monster, d float64) {
		w.damageMonster(m, b.Dmg*(1-d/r*0.5), owner)
		m.Fx.Add(FxBurn, 120)
		hit = true
	})
	w.forAntsNear(b.Pos, r, func(o *Ant, d float64) {
		if w.hostile(b.Colony, o.Colony) {
			w.damageAnt(o, b.Dmg*0.5*(1-d/r*0.5), owner)
			o.Fx.Add(FxBurn, 80)
			hit = true
		}
	})
	if hit && b.Aimed {
		w.AimHits++
	}
	if !hit && owner != nil {
		w.wasteCast(owner)
	}
	w.emit(Event{Kind: EvExplosion, Pos: b.Pos, Colony: b.Colony, Val: r})
}

func (w *World) castFrost(a *Ant) (bool, bool) {
	const r = 85.0
	if _, _, ok := w.nearestFoe(a, r); !ok && !FullBrain() {
		return false, false
	}
	dm := a.Fx.DamageDealt()
	hit := false
	w.forMonstersNear(a.Pos, r, func(m *Monster, d float64) {
		w.damageMonster(m, 5*a.Power*dm, a)
		m.Fx.Add(FxFrost, 200)
		hit = true
	})
	w.forAntsNear(a.Pos, r, func(o *Ant, d float64) {
		if w.hostile(a.Colony, o.Colony) {
			w.damageAnt(o, 3*a.Power*dm, a)
			o.Fx.Add(FxFrost, 160)
			hit = true
		}
	})
	w.emit(Event{Kind: EvFrostNova, Pos: a.Pos, Colony: a.Colony, Val: r})
	return true, hit
}

func (w *World) castStorm(a *Ant) (bool, bool) {
	full := FullBrain()
	if full {
		w.AimShots++
		if _, _, ok := w.foeAhead(a, 160); !ok {
			// Nothing ahead: the bolt crackles into the empty grass.
			end := clampWorld(a.Pos.Add(Polar(a.Angle, 90)))
			w.emit(Event{Kind: EvChain, Pos: a.Pos, Colony: a.Colony, Pts: []Vec2{a.Pos, end}})
			return true, false
		}
		w.AimHits++
	} else if _, _, ok := w.nearestFoe(a, 160); !ok {
		return false, false
	}
	pts := []Vec2{a.Pos}
	var hitM []*Monster // at most 4 jumps: a slice beats a map
	var hitA []*Ant
	seenM := func(m *Monster) bool {
		for _, x := range hitM {
			if x == m {
				return true
			}
		}
		return false
	}
	seenA := func(o *Ant) bool {
		for _, x := range hitA {
			if x == o {
				return true
			}
		}
		return false
	}
	cur := a.Pos
	dmg := 10 * a.Power * a.Fx.DamageDealt()
	for jump := 0; jump < 4; jump++ {
		r := 160.0
		if jump > 0 {
			r = 100
		}
		var bm *Monster
		var ba *Ant
		best := r
		aimed := full && jump == 0 // the first bolt goes where the ant faces
		w.forMonstersNear(cur, r, func(m *Monster, d float64) {
			if aimed && !w.aheadOf(a, m.Pos) {
				return
			}
			if d < best && !seenM(m) {
				bm, ba, best = m, nil, d
			}
		})
		w.forAntsNear(cur, r, func(o *Ant, d float64) {
			if aimed && !w.aheadOf(a, o.Pos) {
				return
			}
			if d < best && w.hostile(a.Colony, o.Colony) && !seenA(o) {
				ba, bm, best = o, nil, d
			}
		})
		switch {
		case bm != nil:
			hitM = append(hitM, bm)
			cur = bm.Pos
			w.damageMonster(bm, dmg, a)
			bm.Fx.Add(FxStun, 25)
		case ba != nil:
			hitA = append(hitA, ba)
			cur = ba.Pos
			w.damageAnt(ba, dmg*0.6, a)
			ba.Fx.Add(FxStun, 20)
		default:
			jump = 99
			continue
		}
		pts = append(pts, cur)
		dmg *= 0.75
	}
	if len(pts) < 2 {
		return false, false
	}
	a.Fx.Add(FxHaste, 180) // the caster rides the discharge
	w.emit(Event{Kind: EvChain, Pos: a.Pos, Colony: a.Colony, Pts: pts})
	return true, true
}

// castEarth lands only if it strikes a foe: shielding sisters with no enemy
// around is spam too.
func (w *World) castEarth(a *Ant) (bool, bool) {
	const r = 75.0
	if _, _, ok := w.nearestFoe(a, r+15); !ok && !FullBrain() {
		return false, false
	}
	hit := false
	w.forAntsNear(a.Pos, r, func(o *Ant, d float64) {
		if w.allied(a.Colony, o.Colony) {
			o.Fx.Add(FxShield, 360)
		} else {
			hit = true
			w.damageAnt(o, 3*a.Power*a.Fx.DamageDealt(), a)
			o.Pos = w.pushOutOfPonds(clampWorld(o.Pos.Add(o.Pos.Sub(a.Pos).Scale(20 / math.Max(d, 1)))))
		}
	})
	w.forMonstersNear(a.Pos, r, func(m *Monster, d float64) {
		w.damageMonster(m, 6*a.Power*a.Fx.DamageDealt(), a)
		hit = true
		if m.Kind != MonCentipede && m.Kind != MonGolem {
			push := m.Pos.Sub(a.Pos)
			m.Pos = clampWorld(m.Pos.Add(push.Scale(24 / math.Max(push.Len(), 1))))
		}
		m.Fx.Add(FxFrost, 40)
	})
	w.emit(Event{Kind: EvQuake, Pos: a.Pos, Colony: a.Colony, Val: r, Elem: ElemEarth})
	return true, hit
}

// castLife lands when the sisters around actually needed care.
func (w *World) castLife(a *Ant) (bool, bool) {
	const r = 95.0
	var targets []Vec2
	need := 0.0
	w.forAntsNear(a.Pos, r, func(o *Ant, d float64) {
		if w.allied(a.Colony, o.Colony) {
			need += 1 - o.HP/o.MaxHP
		}
	})
	if need < 0.3 && !FullBrain() {
		return false, false
	}
	w.forAntsNear(a.Pos, r, func(o *Ant, d float64) {
		if w.hostile(a.Colony, o.Colony) {
			return
		}
		o.Fx.Cleanse()
		heal := math.Min(o.MaxHP-o.HP, 6*a.Power)
		o.HP += heal
		o.Fx.Add(FxRegen, 300)
		a.Healed += heal
		a.Impact += heal
		if len(targets) < 10 && o != a {
			targets = append(targets, o.Pos)
		}
	})
	_ = rand.Float64
	w.emit(Event{Kind: EvHeal, Pos: a.Pos, Colony: a.Colony, Val: r, Pts: targets})
	return true, need >= 0.3
}

// The world's elemental magic: each cast drains its element a little and the
// charge slowly comes back. An element everyone uses runs thin, which gives
// the minority elements their chance (frequency-dependent selection).
const (
	elemDrain    = 0.0025
	elemRecharge = 0.00012
	elemFloor    = 0.1
)

func (w *World) spendElement(e int) {
	w.ElemCharge[e] = math.Max(elemFloor, w.ElemCharge[e]-elemDrain)
}

func (w *World) rechargeElements() {
	for e := range NumElements {
		w.ElemCharge[e] = math.Min(1, w.ElemCharge[e]+elemRecharge)
	}
}
