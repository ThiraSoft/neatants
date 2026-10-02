package sim

import "math"

// chooseRole scores every behaviour from genome traits, senses and two network
// outputs, then keeps the best one (with a little hysteresis to avoid dithering).
func (w *World) chooseRole(a *Ant, o []float64, dNest float64) {
	var food, trail, alarm, threat float64
	for d := range NumSenseDirs {
		b := d * SenseCh
		food = math.Max(food, a.Sense[b])
		trail = math.Max(trail, a.Sense[b+1])
		alarm = math.Max(alarm, a.Sense[b+2])
		threat = math.Max(threat, math.Max(a.Sense[b+3], a.Sense[b+4]))
	}
	hp := a.HP / a.MaxHP
	free := 1 - BoolF(a.Carrying)
	wave := BoolF(w.WaveAlive > 0)
	support := 0.5 + 0.5*BoolF(a.Element == ElemLife || a.Element == ElemEarth)

	var u [NumRoles]float64
	u[RoleForage] = a.Drive * (0.55 + 0.45*math.Max(food, trail)) * free
	u[RoleReturn] = 1.2*BoolF(a.Carrying) + math.Max(0, 0.45-a.Energy)*3 + BoolF(hp < 0.35 && threat == 0)*0.6
	u[RoleFight] = threat*(0.6+a.Courage)*(0.5+0.5*hp)*(0.7+0.3*a.Size)*1.3 + o[6]*0.3*threat + alarm*a.Courage*0.4*free
	u[RoleFlee] = threat * (1 - a.Courage) * (1.2 - hp) * 1.1
	u[RoleGuard] = a.Loyalty * (0.12 + 0.6*wave*math.Max(0, 1-dNest/900)) * free
	u[RoleExplore] = (a.Curiosity*0.7*(1-math.Max(food, trail)) + o[7]*0.25) * free
	u[RoleRescue] = alarm * (0.3 + a.Loyalty) * support * free
	// Raids: bold, not loyal ants of a thriving colony go plunder rivals.
	a.RaidTarget = -1
	col := w.Colonies[a.Colony]
	if w.Tick > 3000 && w.WaveAlive == 0 && col.Pop > 25 {
		best := 1e18
		for _, c := range w.Colonies {
			if c.Alive && w.hostile(a.Colony, c.ID) {
				if d := c.Pos.Sub(a.Pos).Len(); d < best {
					best, a.RaidTarget = d, c.ID
				}
			}
		}
		if a.RaidTarget >= 0 {
			u[RoleRaid] = a.Courage * (1 - a.Loyalty) * math.Min(1, float64(col.Pop)/60) * free * (0.4 + 0.6*a.Size) * 1.1
		}
	}
	u[RoleShop] = w.shopUtility(a, threat)
	if w.Colonies[a.Colony].Streaming(w.Tick) && dNest < 180 && !a.Carrying && !a.Money && !a.Pack {
		u[RoleWatch] = 1.3 * (1 - threat)
	}
	u[a.Role] += 0.12
	a.Utility = u

	best := RoleForage
	for r := range NumRoles {
		if u[r] > u[best] {
			best = Role(r)
		}
	}
	a.Role = best
}

// senseAngle returns the absolute angle of the strongest channel ch, or false.
func (a *Ant) senseAngle(ch int, skipBack bool, floor float64) (float64, bool) {
	bestV, bestD := floor, -1
	for d := range NumSenseDirs {
		if skipBack && d == 3 {
			continue
		}
		if v := a.Sense[d*SenseCh+ch]; v > bestV {
			bestV, bestD = v, d
		}
	}
	if bestD < 0 {
		return 0, false
	}
	return a.Angle + senseDirs[bestD], true
}

func (a *Ant) threatVector() Vec2 {
	var v Vec2
	for d := range NumSenseDirs {
		b := d * SenseCh
		t := math.Max(a.Sense[b+3], a.Sense[b+4])
		v = v.Add(Polar(a.Angle+senseDirs[d], t))
	}
	return v
}

// roleSteer converts the chosen role into a desired heading.
func (w *World) roleSteer(a *Ant, toNest Vec2) (float64, bool) {
	nestAng := math.Atan2(toNest.Y, toNest.X)
	dNest := toNest.Len()
	switch a.Role {
	case RoleReturn:
		return nestAng, true
	case RoleForage:
		if ang, ok := a.senseAngle(0, false, 0); ok {
			return ang, true
		}
		return a.senseAngle(1, true, 0.05)
	case RoleFight:
		tv := a.threatVector()
		if tv.Len() > 0.01 {
			return math.Atan2(tv.Y, tv.X), true
		}
		return a.senseAngle(2, false, 0.1)
	case RoleFlee:
		tv := a.threatVector().Scale(-1).Add(toNest.Scale(0.6 / math.Max(dNest, 1)))
		return math.Atan2(tv.Y, tv.X), true
	case RoleGuard:
		tv := a.threatVector()
		if tv.Len() > 0.2 && dNest < 260 {
			return math.Atan2(tv.Y, tv.X), true
		}
		if dNest > 170 {
			return nestAng, true
		}
		dir := 1.0
		if a.ID%2 == 0 {
			dir = -1
		}
		return nestAng + dir*(math.Pi/2+ClampF((dNest-110)/80, -0.6, 0.6)), true
	case RoleExplore:
		return nestAng + math.Pi + math.Sin(a.wander*0.4)*0.8, true
	case RoleRescue:
		return a.senseAngle(2, false, 0.05)
	case RoleShop:
		return w.shopSteer(a, toNest)
	case RoleWatch:
		if dNest > 60 {
			return nestAng, true
		}
		return nestAng, true // face the queen
	case RoleRaid:
		if a.RaidTarget >= 0 {
			to := w.Colonies[a.RaidTarget].Pos.Sub(a.Pos)
			tv := a.threatVector()
			if tv.Len() > 0.3 {
				return math.Atan2(tv.Y, tv.X), true
			}
			return math.Atan2(to.Y, to.X), true
		}
	}
	return 0, false
}
