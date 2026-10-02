package sim

import (
	"fmt"
	"github.com/ThiraSoft/neatants/neat"
	"math"
	"math/rand"
)

// Colony marriages. Two neighbouring colonies that have left each other alone
// for long enough may unite: no more fighting, shared pheromone trails, common
// defence, a dowry of food, and a half-blood queen if there is room on the
// map. Accidents happen (insecticide sprays hit everyone): enough wounds and
// the couple divorces.

// allied reports whether ants of colonies a and b are on the same side.
func (w *World) allied(a, b int) bool {
	return a == b || (w.Colonies[a].Spouse == b && w.Colonies[b].Spouse == a)
}

func (w *World) hostile(a, b int) bool { return !w.allied(a, b) }

// woundHook lets tests observe wounds between spouses.
var woundHook func(att, vic int)

func (w *World) SpouseOf(c int) int {
	s := w.Colonies[c].Spouse
	if s >= 0 && s < len(w.Colonies) && w.Colonies[s].Alive && w.Colonies[s].Spouse == c {
		return s
	}
	return -1
}

// hostileAct records an attack from colony att on colony vic. Between
// rivals only serious acts (a kill, a raid) break the peace; between spouses
// every wound is a grievance, and too many of them end the marriage.
func (w *World) hostileAct(att, vic int, serious bool) {
	if att == vic {
		return
	}
	if !w.allied(att, vic) {
		if serious {
			w.lastFight[att][vic], w.lastFight[vic][att] = w.Tick, w.Tick
		}
		return
	}
	if woundHook != nil {
		woundHook(att, vic)
	}
	w.grievance(w.Colonies[att], w.Colonies[vic], "after one insecticide blow too many")
}

// grievance adds a grudge to a couple; past the limit, they divorce.
func (w *World) grievance(a, b *Colony, reason string) {
	a.Grief++
	b.Grief++
	if a.Grief >= Cfg.DivorceGrief {
		w.divorce(a, b, reason)
	}
}

func (w *World) Marry(a, b *Colony) {
	a.Spouse, b.Spouse = b.ID, a.ID
	a.Grief, b.Grief = 0, 0
	a.MarriedAt, b.MarriedAt = w.Tick, w.Tick
	mid := LerpV(a.Pos, b.Pos, 0.5)
	w.WeddingArch[a.ID], w.WeddingArch[b.ID] = mid, mid
	w.log(fmt.Sprintf("Wedding! %s and %s join their colonies.", a.Name, b.Name), [3]float64{1, 0.55, 0.75})
	w.emit(Event{Kind: EvWedding, Pos: mid, Colony: a.ID, Val: float64(b.ID), Pts: []Vec2{a.Pos, b.Pos}})
	w.halfBloodQueen(a, b, mid)
}

func (w *World) divorce(a, b *Colony, reason string) {
	a.Spouse, b.Spouse = -1, -1
	a.Grief, b.Grief = 0, 0
	a.GiftStreak, b.GiftStreak = 0, 0
	a.Abandon, b.Abandon = 0, 0
	w.lastFight[a.ID][b.ID], w.lastFight[b.ID][a.ID] = w.Tick, w.Tick
	w.log(fmt.Sprintf("Divorce! %s and %s split up, %s.", a.Name, b.Name, reason), [3]float64{0.75, 0.7, 0.8})
	w.emit(Event{Kind: EvDivorce, Pos: LerpV(a.Pos, b.Pos, 0.5), Colony: a.ID, Val: float64(b.ID)})
}

// nestAttacked: a colony whose nest is hit while no ant of its spouse comes
// to help ends up feeling abandoned.
func (w *World) nestAttacked(c *Colony) {
	s := w.SpouseOf(c.ID)
	if s < 0 {
		return
	}
	helpers := 0
	w.forAntsNear(c.Pos, 450, func(a *Ant, d float64) {
		if a.Colony == s {
			helpers++
		}
	})
	if helpers > 0 {
		return
	}
	c.Abandon++
	if c.Abandon >= 15 {
		c.Abandon = 0
		w.grievance(c, w.Colonies[s], "abandoned during a monster attack")
	}
}

// widow is called when a colony falls: its spouse is left alone.
func (w *World) widow(c *Colony) {
	s := w.SpouseOf(c.ID)
	c.Spouse = -1
	if s < 0 {
		return
	}
	w.Colonies[s].Spouse = -1
	w.log(fmt.Sprintf("%s is widowed of %s.", w.Colonies[s].Name, c.Name), w.Colonies[s].Color)
}

// updateMarriages looks for couples, shares food between spouses and lets
// old grievances fade.
func (w *World) updateMarriages() {
	if !Cfg.Marriage || w.Tick%300 != 0 {
		return
	}
	for _, c := range w.Colonies {
		if w.Tick%3600 == 0 && c.Grief > 0 {
			c.Grief-- // time heals
		}
	}
	for i, a := range w.Colonies {
		for _, b := range w.Colonies[i+1:] {
			if !a.Alive || !b.Alive || a.Spouse >= 0 || b.Spouse >= 0 || a.Pop < 10 || b.Pop < 10 {
				continue
			}
			if a.Pos.Sub(b.Pos).Len() > 2000 {
				continue
			}
			since := w.Tick - max(w.lastFight[a.ID][b.ID], a.Founded, b.Founded)
			if since >= Cfg.MarriageDelay && rand.Float64() < 0.3 {
				w.Marry(a, b)
			}
		}
	}
	// The richer spouse shares part of its reserves.
	for _, a := range w.Colonies {
		s := w.SpouseOf(a.ID)
		if s < 0 || a.ID > s {
			continue
		}
		b := w.Colonies[s]
		rich, poor := a, b
		if b.Food > a.Food {
			rich, poor = b, a
		}
		if d := rich.Food - poor.Food; d > 10 {
			gift := d * 0.2
			w.lose(rich, gift)
			w.earn(poor, gift)
			w.emit(Event{Kind: EvGift, Pos: poor.Pos, Pts: []Vec2{rich.Pos}, Colony: rich.ID})
			// Always giving while short yourself breeds resentment.
			rich.GiftStreak++
			poor.GiftStreak = 0
			if rich.GiftStreak >= 3 && rich.Food < 25 {
				rich.GiftStreak = 0
				w.grievance(rich, poor, "tired of feeding the in-laws")
			}
		}
	}
}

// halfBloodQueen founds a new colony between the spouses from crossed
// genomes of both lineages, if a slot is free on the map.
func (w *World) halfBloodQueen(a, b *Colony, mid Vec2) {
	slot := -1
	for _, c := range w.Colonies {
		if !c.Alive {
			slot = c.ID
			break
		}
	}
	if slot < 0 && len(w.Colonies) < MaxColonies {
		slot = len(w.Colonies)
	}
	if slot < 0 || len(a.Hall) == 0 || len(b.Hall) == 0 {
		return
	}
	var pool []*neat.Genome
	for range 12 {
		ga, gb := weightedPick(a.Hall), weightedPick(b.Hall)
		if gb.Fitness > ga.Fitness {
			ga, gb = gb, ga
		}
		child := neat.Crossover(ga, gb, 0)
		child.Mutate()
		pool = append(pool, child)
	}
	// Settle beside the wedding arch, away from ponds and other nests.
	d := b.Pos.Sub(a.Pos)
	perp := Vec2{-d.Y, d.X}.Scale(1 / math.Max(d.Len(), 1))
	pos := mid
	for _, off := range []float64{320, -320, 450, -450, 200, -200} {
		p := clampWorld(mid.Add(perp.Scale(off)))
		ok := !w.InPond(p, 90) && p.Sub(w.Cave).Len() > 600 && p.Sub(w.Shop).Len() > 300
		for _, c := range w.Colonies {
			ok = ok && (!c.Alive || p.Sub(c.Pos).Len() > 350)
		}
		if ok {
			pos = p
			break
		}
	}
	child := w.foundColony(slot, pos, pool)
	delete(w.usedNames, child.Name)
	child.Name = halfName(a.Name, b.Name)
	w.usedNames[child.Name] = true
	child.Color = Mixc(a.Color, b.Color, 0.5)
	w.log(fmt.Sprintf("A half-blood queen is born of %s and %s: the colony %s.", a.Name, b.Name, child.Name), child.Color)
}

// halfName blends two colony names: "Braise" + "Azur" -> "Brazur".
func halfName(a, b string) string {
	ra, rb := []rune(a), []rune(b)
	return string(ra[:(len(ra)+1)/2]) + string(rb[len(rb)/2:])
}
