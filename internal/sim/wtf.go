package sim

import (
	"fmt"
	"math"
	"math/rand"
)

// The silly side of the meadow: the Wallmart security guard, the scooter
// courier, absurd weather (frog rain, confetti storm) and influencer queens.

// --- Security guard ---

type Guard struct {
	Pos, Home      Vec2
	Angle, Heading float64
	Gait           float64
	Target         *Ant
	CD             int
	Patrol         float64
	Catches        int
}

const guardLeash = 520.0

func (w *World) updateGuard() {
	g := w.Guard
	if g == nil {
		return
	}
	if g.CD > 0 {
		g.CD--
	}
	if t := g.Target; t != nil && (!t.Alive || !t.Stolen || t.Pos.Sub(w.Shop).Len() > guardLeash) {
		g.Target = nil
	}
	if g.Target == nil {
		w.forAntsNear(g.Pos, 170, func(a *Ant, d float64) {
			if a.Stolen && g.Target == nil {
				g.Target = a
			}
		})
	}
	var goal Vec2
	speed := 0.9
	if t := g.Target; t != nil {
		goal, speed = t.Pos, 2.3
		if t.Pos.Sub(g.Pos).Len() < 12 && g.CD == 0 {
			g.CD = 40
			w.damageAnt(t, 5, nil)
			w.emit(Event{Kind: EvGuardBite, Pos: t.Pos})
			if !t.Alive {
				g.Catches++
				g.Target = nil
			}
		}
	} else {
		// Stroll around the parking lot.
		g.Patrol += 0.004
		goal = w.Shop.Add(Vec2{math.Cos(g.Patrol) * ShopW * 0.7, math.Sin(g.Patrol) * ShopH * 0.75})
	}
	to := goal.Sub(g.Pos)
	if to.Len() > 2 {
		want := math.Atan2(to.Y, to.X)
		g.Angle += ClampF(angDiff(want, g.Angle), -0.12, 0.12)
		g.Pos = g.Pos.Add(Polar(g.Angle, speed))
		g.Gait += speed * 0.4
	}
	g.Heading += angDiff(g.Angle, g.Heading) * 0.2
	g.Pos = w.pushOutOfPonds(clampWorld(g.Pos))
}

// --- Scooter courier ---

type Courier struct {
	Pos, Prev Vec2
	Angle     float64
	Colony    int
	Back      bool
	Alive     bool
	lastHit   int
}

const courierSpeed = 5.0

// maybeOrderDelivery: a rich colony pays extra to get its pack delivered.
func (w *World) maybeOrderDelivery(c *Colony) {
	if !Cfg.ShopEnabled || w.ShopStock == 0 || c.Pop < 25 || !w.wantsInsecticide(c) {
		return
	}
	cost := w.ShopPrice() + Cfg.DeliveryFee
	if c.Food < cost+3 || rand.Float64() > 0.004 {
		return
	}
	for _, k := range w.Couriers {
		if k.Alive && k.Colony == c.ID {
			return
		}
	}
	c.Food -= cost
	w.ShopStock--
	to := c.Pos.Sub(w.Shop)
	w.Couriers = append(w.Couriers, &Courier{Pos: w.Shop, Prev: w.Shop, Angle: math.Atan2(to.Y, to.X), Colony: c.ID, Alive: true})
	if w.Tick-w.lastDeliveryLog > 1800 {
		w.lastDeliveryLog = w.Tick
		w.log(c.Name+" has the insecticide delivered by scooter.", c.Color)
	}
}

func (w *World) updateCouriers() {
	live := w.Couriers[:0]
	for _, k := range w.Couriers {
		if !k.Alive {
			continue
		}
		col := w.Colonies[k.Colony]
		goal := col.Pos
		if k.Back || !col.Alive {
			goal, k.Back = w.Shop, true
		}
		to := goal.Sub(k.Pos)
		if to.Len() < NestRadius {
			if k.Back {
				continue // parked back at the Wallmart
			}
			col.Cans += Cfg.InsecticidePack
			k.Back = true
			w.emit(Event{Kind: EvCourierArrive, Pos: k.Pos, Colony: k.Colony})
		}
		want := math.Atan2(to.Y, to.X)
		k.Angle += ClampF(angDiff(want, k.Angle), -0.1, 0.1)
		k.Prev = k.Pos
		k.Pos = w.pushOutOfPonds(clampWorld(k.Pos.Add(Polar(k.Angle, courierSpeed))))
		// Anyone in the way gets run over, customers included.
		w.forAntsNear(k.Pos, 8, func(a *Ant, d float64) {
			w.damageAnt(a, 8, nil)
			if w.Tick-k.lastHit > 20 {
				k.lastHit = w.Tick
				w.emit(Event{Kind: EvCourierHit, Pos: a.Pos})
			}
		})
		live = append(live, k)
	}
	clear(w.Couriers[len(live):])
	w.Couriers = live
}

// --- Absurd weather ---

const (
	WeirdNone = iota
	WeirdFrogs
	WeirdConfetti
)

type Frog struct {
	Pos    Vec2
	Z, VZ  float64
	Hop    int
	From   Vec2 // hop start, for the arc
	To     Vec2
	HopT   float64
	Life   int
	CD     int
	Tongue float64 // tongue animation, 1 = fully out
	Lick   Vec2
	Alive  bool
}

func (w *World) updateWeird() {
	if !Cfg.WeirdWeather {
		return
	}
	switch {
	case w.Weird == WeirdNone && w.Tick >= w.NextWeird:
		w.Weird = WeirdFrogs + rand.Intn(2)
		w.WeirdUntil = w.Tick + 1500 + rand.Intn(900)
		if w.Weird == WeirdFrogs {
			w.log("It is raining frogs! They eat everything that moves.", [3]float64{0.5, 0.95, 0.4})
		} else {
			w.log("A confetti storm scrambles every trail.", [3]float64{1, 0.6, 0.9})
		}
		w.emit(Event{Kind: EvWeirdStart, Val: float64(w.Weird)})
	case w.Weird != WeirdNone && w.Tick >= w.WeirdUntil:
		w.Weird = WeirdNone
		w.NextWeird = w.Tick + int(Cfg.DayLength*60*(0.8+rand.Float64()*0.8))
	}

	if w.Weird == WeirdFrogs && w.Tick%12 == 0 {
		// Half of them rain down where the ants are.
		p := Vec2{rand.Float64() * WorldW, rand.Float64() * WorldH}
		if len(w.Ants) > 0 && rand.Float64() < 0.5 {
			p = clampWorld(w.Ants[rand.Intn(len(w.Ants))].Pos.Add(Polar(rand.Float64()*6.28, 60+rand.Float64()*140)))
		}
		w.Frogs = append(w.Frogs, &Frog{Pos: p, Z: 400, Life: 1000, Alive: true, Hop: 40})
	}
	if w.Weird == WeirdConfetti {
		// Scramble the pheromones: wipe real trails, sprinkle fake ones.
		if w.Tick%4 == 0 {
			for c := range len(w.Colonies) {
				t := w.Trail[c]
				for i := range t {
					t[i] *= 0.9
				}
			}
		}
		for range 25 {
			c := rand.Intn(len(w.Colonies))
			w.Trail[c][rand.Intn(Cells)] = float32(0.5 + rand.Float64()*0.5)
		}
	}
	w.updateFrogs()
}

func (w *World) updateFrogs() {
	live := w.Frogs[:0]
	for _, f := range w.Frogs {
		f.Life--
		if f.Life <= 0 {
			continue
		}
		if f.CD > 0 {
			f.CD--
		}
		f.Tongue *= 0.8
		if f.Z > 0 { // still falling from the sky
			f.VZ -= 0.35
			f.Z = math.Max(0, f.Z+f.VZ)
			if f.Z == 0 {
				w.emit(Event{Kind: EvFrogLand, Pos: f.Pos})
				w.forAntsNear(f.Pos, 10, func(a *Ant, d float64) { w.damageAnt(a, 999, nil) })
			}
			live = append(live, f)
			continue
		}
		if f.HopT > 0 { // mid-hop
			f.HopT = math.Max(0, f.HopT-0.05)
			f.Pos = LerpV(f.To, f.From, f.HopT)
			live = append(live, f)
			continue
		}
		// Look for a snack, ant or monster alike.
		var prey Vec2
		found, best := false, 170.0
		var ant *Ant
		var mon *Monster
		w.forAntsNear(f.Pos, best, func(a *Ant, d float64) {
			if d < best {
				best, prey, found, ant, mon = d, a.Pos, true, a, nil
			}
		})
		w.forMonstersNear(f.Pos, best, func(m *Monster, d float64) {
			if d < best {
				best, prey, found, mon, ant = d, m.Pos, true, m, nil
			}
		})
		if found && best < 26 && f.CD == 0 {
			f.CD, f.Tongue, f.Lick = 80, 1, prey
			if ant != nil {
				w.damageAnt(ant, 999, nil)
			} else if mon != nil {
				w.damageMonster(mon, 25, nil)
			}
			w.emit(Event{Kind: EvFrogEat, Pos: prey, Pts: []Vec2{f.Pos}})
		} else if f.Hop--; f.Hop <= 0 {
			f.Hop = 30 + rand.Intn(40)
			dir := rand.Float64() * 2 * math.Pi
			if found {
				to := prey.Sub(f.Pos)
				dir = math.Atan2(to.Y, to.X)
			}
			f.From = f.Pos
			f.To = w.pushOutOfPonds(clampWorld(f.Pos.Add(Polar(dir, math.Min(28, best)))))
			f.HopT = 1
		}
		live = append(live, f)
	}
	clear(w.Frogs[len(live):])
	w.Frogs = live
}

// --- Influencer queens ---

// updateLive: now and then a queen goes live from the nest. Nearby workers
// stop to watch (and get a morale boost), which costs the colony their work.
func (w *World) updateLive(c *Colony) {
	if !Cfg.QueenLive {
		return
	}
	if c.NextLive == 0 {
		c.NextLive = w.Tick + int(Cfg.DayLength*60*(0.5+rand.Float64()))
	}
	if c.LiveUntil <= w.Tick && w.Tick >= c.NextLive && c.Pop >= 15 && w.WaveAlive == 0 {
		c.LiveUntil = w.Tick + 900
		c.NextLive = w.Tick + int(Cfg.DayLength*60*(0.8+rand.Float64()))
		c.Likes = 0
		w.log(fmt.Sprintf("The queen of %s starts a livestream from the nest!", c.Name), c.Color)
		w.emit(Event{Kind: EvLiveStart, Pos: c.Pos, Colony: c.ID})
	}
	live := c.LiveUntil > w.Tick
	ending := c.LiveUntil == w.Tick
	if !live && !ending {
		return
	}
	viewers := 0
	w.forAntsNear(c.Pos, 220, func(a *Ant, d float64) {
		if a.Colony != c.ID {
			return
		}
		if ending {
			a.Fx.Add(FxMorale, 1800) // the vibe lasts after the stream
		} else {
			a.Fx.Add(FxMorale, 30)
			if a.Role == RoleWatch {
				viewers++
			}
		}
	})
	if w.Tick%60 == 0 {
		c.Likes += viewers * (3 + rand.Intn(20))
	}
	if ending {
		w.log(fmt.Sprintf("End of the live by %s: %s likes.", c.Name, FmtLikes(c.Likes)), c.Color)
	}
}

func (c *Colony) Streaming(tick int) bool { return c.LiveUntil > tick }

func FmtLikes(n int) string {
	if n >= 1000 {
		return fmt.Sprintf("%.1fk", float64(n)/1000)
	}
	return fmt.Sprint(n)
}
