package sim

// Buffs and debuffs shared by ants and monsters. Each effect is a countdown in
// ticks; applying an effect keeps the longer of the current and new duration.

type EffectKind uint8

const (
	// Buffs
	FxShield EffectKind = iota // damage taken x0.3
	FxRegen                    // heals over time
	FxHaste                    // speed x1.35
	FxMight                    // damage dealt x1.5
	FxMorale                   // queen's live stream: damage x1.2, speed x1.1
	// Debuffs
	FxBurn   // damage over time
	FxFrost  // speed x0.45
	FxStun   // cannot move, bite or cast
	FxPoison // damage over time, halves healing
	FxWeak   // damage dealt x0.6
	FxOld    // old age: slower and weaker (set from age, not a timer)
	FxPlague // cave fever: contagious among monsters, eats a share of max HP
	NumEffects
)

type effectInfo struct {
	Name  string
	Buff  bool
	Color [3]float64
}

var Effect = [NumEffects]effectInfo{
	FxShield: {"Shield", true, [3]float64{0.95, 0.78, 0.4}},
	FxRegen:  {"Regeneration", true, [3]float64{0.45, 1, 0.5}},
	FxHaste:  {"Haste", true, [3]float64{0.55, 0.95, 1}},
	FxMight:  {"Frenzy", true, [3]float64{1, 0.35, 0.3}},
	FxMorale: {"High spirits", true, [3]float64{1, 0.55, 0.8}},
	FxBurn:   {"Burn", false, [3]float64{1, 0.5, 0.15}},
	FxFrost:  {"Freeze", false, [3]float64{0.55, 0.85, 1}},
	FxStun:   {"Stunned", false, [3]float64{1, 0.95, 0.4}},
	FxPoison: {"Poison", false, [3]float64{0.55, 0.9, 0.2}},
	FxWeak:   {"Weakness", false, [3]float64{0.7, 0.5, 0.85}},
	FxOld:    {"Old age", false, [3]float64{0.75, 0.72, 0.68}},
	FxPlague: {"Cave fever", false, [3]float64{0.78, 0.88, 0.3}},
}

type Effects [NumEffects]int32

func (e *Effects) Add(k EffectKind, ticks int) {
	if int32(ticks) > e[k] {
		e[k] = int32(ticks)
	}
}

func (e *Effects) Has(k EffectKind) bool { return e[k] > 0 }

// Tick counts every timed effect down by one.
func (e *Effects) Tick() {
	for i := range e {
		if EffectKind(i) != FxOld && e[i] > 0 {
			e[i]--
		}
	}
}

// Cleanse removes every debuff except old age.
func (e *Effects) Cleanse() {
	for i := range e {
		if k := EffectKind(i); !Effect[k].Buff && k != FxOld {
			e[i] = 0
		}
	}
}

func (e *Effects) SpeedMul() float64 {
	if e.Has(FxStun) {
		return 0
	}
	m := 1.0
	if e.Has(FxFrost) {
		m *= 0.45
	}
	if e.Has(FxHaste) {
		m *= 1.35
	}
	if e.Has(FxMorale) {
		m *= 1.1
	}
	if e.Has(FxOld) {
		m *= 0.8
	}
	if e.Has(FxPlague) {
		m *= 0.8
	}
	return m
}

func (e *Effects) DamageDealt() float64 {
	m := 1.0
	if e.Has(FxMight) {
		m *= 1.5
	}
	if e.Has(FxMorale) {
		m *= 1.2
	}
	if e.Has(FxWeak) {
		m *= 0.6
	}
	if e.Has(FxOld) {
		m *= 0.85
	}
	if e.Has(FxPlague) {
		m *= 0.8
	}
	return m
}

func (e *Effects) DamageTaken() float64 {
	if e.Has(FxShield) {
		return 0.3
	}
	return 1
}

func (e *Effects) HealMul() float64 {
	if e.Has(FxPoison) {
		return 0.5
	}
	return 1
}

// Count returns how many buffs and debuffs are active.
func (e *Effects) Count() (buffs, debuffs int) {
	for i, t := range e {
		if t > 0 {
			if Effect[i].Buff {
				buffs++
			} else {
				debuffs++
			}
		}
	}
	return
}

// dotDamage returns the damage-over-time owed this tick for a creature,
// scaled by `scale` (ants and monsters have very different HP pools).
func (e *Effects) dotDamage(tick int, scale float64) float64 {
	d := 0.0
	if e.Has(FxBurn) && tick%20 == 0 {
		d += 0.4 * scale
	}
	if e.Has(FxPoison) && tick%15 == 0 {
		d += 0.3 * scale
	}
	return d
}
