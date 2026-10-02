package render

import (
	"github.com/ThiraSoft/neatants/internal/sim"
	"math"
	"math/rand"
)

// fxTint colours a creature's body according to its debuffs.
func fxTint(c [3]float64, e *sim.Effects) [3]float64 {
	if e.Has(sim.FxOld) {
		c = sim.Mixc(c, [3]float64{0.5, 0.48, 0.46}, 0.35)
	}
	if e.Has(sim.FxFrost) {
		c = sim.Mixc(c, [3]float64{0.55, 0.82, 1}, 0.5)
	}
	if e.Has(sim.FxBurn) {
		c = sim.Mixc(c, [3]float64{1, 0.4, 0.1}, 0.35)
	}
	if e.Has(sim.FxPoison) {
		c = sim.Mixc(c, [3]float64{0.45, 0.85, 0.15}, 0.35)
	}
	if e.Has(sim.FxWeak) {
		c = sim.Mixc(c, [3]float64{0.45, 0.3, 0.55}, 0.3)
	}
	if e.Has(sim.FxPlague) {
		c = sim.Mixc(c, [3]float64{0.55, 0.65, 0.2}, 0.4)
	}
	return c
}

// drawAuras draws the per-frame visuals of active effects around a creature
// centred on screen point (x, y) with screen radius rad and heading h.
func (r *Renderer) drawAuras(e *sim.Effects, x, y, rad, h float64, seed int) {
	t := r.time + float64(seed)*0.37
	pulse := float32(0.75 + 0.25*math.Sin(t*5))
	if e.Has(sim.FxMight) {
		r.fxB.Glow(x, y, rad*2.2, 1, 0.25, 0.15, 0.35*pulse)
		r.lightB.Glow(x, y, rad*6, 1, 0.3, 0.15, 0.25)
	}
	if e.Has(sim.FxRegen) {
		r.fxB.Glow(x, y, rad*1.8, 0.4, 1, 0.45, 0.16*pulse)
	}
	if e.Has(sim.FxWeak) {
		r.fxB.Glow(x, y, rad*1.6, 0.5, 0.3, 0.7, 0.18)
	}
	if e.Has(sim.FxHaste) {
		fx, fy := math.Cos(h), math.Sin(h)
		for k := 1; k <= 3; k++ {
			d := float64(k) * rad * 0.9
			r.fxB.Line(SprGlow, x-fx*d, y-fy*d, x-fx*(d+rad*0.8), y-fy*(d+rad*0.8), rad*0.9,
				c4(0.55, 0.95, 1, 0.3/float64(k)))
		}
	}
	if e.Has(sim.FxShield) {
		fa := float32(math.Min(1, float64(e[sim.FxShield])/40))
		r.fxB.Quad(SprBubble, x, y, rad*1.6, rad*1.6, 0, 1, 0.8, 0.4, 0.4*fa)
	}
	if e.Has(sim.FxPlague) {
		// A small cloud of flies buzzing around the sick creature.
		for k := range 5 {
			a := t*9 + float64(k)*1.26 + math.Sin(t*13+float64(k))*0.8
			d := rad * (1.1 + 0.3*math.Sin(t*7+float64(k)*2))
			r.topB.Circle(x+math.Cos(a)*d, y+math.Sin(a)*d*0.7-rad*0.4, math.Max(0.8, rad*0.07), c4(0.08, 0.08, 0.06, 0.9))
		}
		r.fxB.Glow(x, y, rad*1.7, 0.7, 0.85, 0.25, 0.12*pulse)
	}
	if e.Has(sim.FxStun) {
		for k := range 3 {
			a := t*5 + float64(k)*2.09
			sx := x + math.Cos(a)*rad*0.9
			sy := y - rad*1.3 + math.Sin(a)*rad*0.3
			r.fxB.Quad(SprStar, sx, sy, rad*0.45, rad*0.45, a, 1, 0.95, 0.45, 0.95)
		}
	}
}

// effectParticles spawns the ambient particles of active effects.
func (r *Renderer) effectParticles(e *sim.Effects, p sim.Vec2, radius, rate float64) {
	emit := func(k sim.EffectKind) bool { return e.Has(k) && rand.Float64() < rate }
	at := func() sim.Vec2 { return p.Add(sim.Polar(rnd(0, 6.28), rnd(0, radius))) }
	if emit(sim.FxBurn) {
		q := at()
		r.parts.Add(Particle{X: q.X, Y: q.Y, VZ: rnd(0.4, 1), Max: 30, Size: rnd(1.5, 3.5), Grow: -0.07, R: 1, G: 0.7, B: 0.2, R2: 1, G2: 0.2, B2: 0, A: 1, Spr: SprGlow, Add: true, Light: 0.15})
	}
	if emit(sim.FxFrost) {
		q := at()
		r.parts.Add(Particle{X: q.X, Y: q.Y, VZ: rnd(0.1, 0.3), Max: 40, Size: rnd(1.2, 2.6), R: 0.7, G: 0.95, B: 1, A: 0.9, Spr: SprStar, Add: true, Rot: rnd(0, 1)})
	}
	if emit(sim.FxPoison) {
		q := at()
		r.parts.Add(Particle{X: q.X, Y: q.Y, VZ: rnd(0.3, 0.6), Max: 45, Size: rnd(0.8, 1.8), Grow: 0.03, R: 0.55, G: 0.95, B: 0.25, A: 0.8, Spr: SprRing})
	}
	if emit(sim.FxRegen) {
		q := at()
		r.parts.Add(Particle{X: q.X, Y: q.Y, VZ: rnd(0.3, 0.7), Max: 50, Size: rnd(1, 2), R: 0.55, G: 1, B: 0.6, A: 0.9, Spr: SprStar, Add: true})
	}
	if emit(sim.FxMight) {
		q := at()
		r.parts.Add(Particle{X: q.X, Y: q.Y, VZ: rnd(0.5, 1.1), Max: 30, Size: rnd(0.8, 1.6), R: 1, G: 0.35, B: 0.25, A: 1, Spr: SprGlow, Add: true, Stretch: 1})
	}
	if emit(sim.FxWeak) {
		q := at()
		r.parts.Add(Particle{X: q.X, Y: q.Y, VZ: rnd(0.1, 0.3), Max: 60, Size: rnd(3, 6), Grow: 0.05, R: 0.35, G: 0.2, B: 0.45, A: 0.35, Spr: SprSmoke, Rot: rnd(0, 6)})
	}
	if emit(sim.FxPlague) {
		q := at()
		r.parts.Add(Particle{X: q.X, Y: q.Y, Z: rnd(2, 6), VX: rnd(-0.1, 0.1), VY: rnd(-0.1, 0.1), Grav: 0.05, Max: 50,
			Size: rnd(0.8, 1.6), R: 0.7, G: 0.82, B: 0.25, A: 0.9, Spr: SprDot, Decal: SprSplat})
		if rand.Float64() < 0.4 {
			r.parts.Add(Particle{X: q.X, Y: q.Y, VZ: rnd(0.05, 0.2), Max: 90, Size: rnd(4, 8), Grow: 0.08,
				R: 0.6, G: 0.7, B: 0.25, A: 0.25, Spr: SprSmoke, Rot: rnd(0, 6)})
		}
	}
	if e.Has(sim.FxOld) && rand.Float64() < rate*0.3 {
		q := at()
		r.parts.Add(Particle{X: q.X, Y: q.Y, VX: rnd(-0.2, 0.2), VY: rnd(-0.2, 0.2), Max: 50, Size: rnd(0.6, 1.2), R: 0.8, G: 0.78, B: 0.72, A: 0.5, Spr: SprSoft})
	}
}

func (r *Renderer) oldAgeDeath(p sim.Vec2, cc [3]float32, starved bool) {
	c := [3]float32{(cc[0] + 1) / 2, (cc[1] + 1) / 2, (cc[2] + 0.9) / 2}
	if starved {
		c = [3]float32{0.6, 0.58, 0.55}
	}
	r.parts.Add(Particle{X: p.X, Y: p.Y, Z: 2, VZ: 0.22, VX: rnd(-0.05, 0.05), Max: 170, Size: 3, Grow: -0.008,
		R: c[0], G: c[1], B: c[2], A: 0.8, Spr: SprGlow, Add: true, Light: 0.2})
	if !starved {
		for range 5 {
			r.parts.Add(Particle{X: p.X + rnd(-4, 4), Y: p.Y + rnd(-4, 4), VZ: rnd(0.15, 0.4), Max: rnd(60, 110), Size: rnd(0.8, 1.5),
				R: 1, G: 0.9, B: 0.6, A: 0.9, Spr: SprStar, Add: true, Rot: rnd(0, 1)})
		}
	}
	r.decal(SprSplat, p, 4, rnd(0, 6), [4]float32{0.1, 0.08, 0.06, 0.2})
}

func (r *Renderer) monsterOldDeath(p sim.Vec2, k sim.MonsterKind) {
	rad := sim.MonsterSpecs[k].Radius
	for range 10 + int(rad) {
		q := p.Add(sim.Polar(rnd(0, 6.28), rnd(0, rad)))
		r.parts.Add(Particle{X: q.X, Y: q.Y, VX: rnd(-0.4, 0.4), VY: rnd(-0.4, 0.4), VZ: rnd(0, 0.3), Drag: 0.02, Max: rnd(70, 130),
			Size: rnd(4, 9), Grow: 0.12, R: 0.45, G: 0.42, B: 0.4, A: 0.45, Spr: SprSmoke, Rot: rnd(0, 6)})
	}
	for range 8 {
		q := p.Add(sim.Polar(rnd(0, 6.28), rnd(0, rad)))
		r.parts.Add(Particle{X: q.X, Y: q.Y, Z: rnd(2, 8), VX: rnd(-0.6, 0.6), VY: rnd(-0.6, 0.6), Grav: 0.08, Max: 60,
			Size: rnd(1, 2), R: 0.3, G: 0.27, B: 0.25, A: 1, Spr: SprDot})
	}
	r.decal(SprSplat, p, rad*1.6, rnd(0, 6), [4]float32{0.2, 0.19, 0.18, 0.35})
}

// contagion sends a drift of spores from the sick monster to its new victim.
func (r *Renderer) contagion(from, to sim.Vec2) {
	d := to.Sub(from)
	for k := range 7 {
		f := float64(k) / 7
		q := from.Add(d.Scale(f * 0.3))
		r.parts.Add(Particle{X: q.X, Y: q.Y, Z: 6, VX: d.X * 0.012, VY: d.Y * 0.012, Max: 55, Size: rnd(1.5, 3), Grow: 0.02,
			R: 0.75, G: 0.9, B: 0.3, A: 0.55, Spr: SprGlow, Add: true})
	}
}

// rect draws an axis-aligned filled rectangle centred on (x, y).
func (b *Batch) rect(x, y, w, h float64, c col4) {
	b.Quad(SprSolid, x, y, w/2, h/2, 0, c[0], c[1], c[2], c[3])
}

// drawShop renders the Wallmart: parking lot, building, lamps at night.
func (r *Renderer) drawShop(w *sim.World, shx, shy float64) {
	if !sim.Cfg.ShopEnabled || !r.visible(w.Shop, sim.ShopW) {
		return
	}
	z := r.cam.Z
	x, y := r.ws(w.Shop)
	W, H := sim.ShopW*z, sim.ShopH*z
	night := float32(r.night)

	// Parking lot with painted bays
	r.shadowB.rect(x+shx*3*z, y+shy*3*z, W+6*z, H+6*z, c4(0, 0, 0, 0.3))
	r.bushB.rect(x, y, W, H, c4(0.24, 0.24, 0.26, 1))
	r.bushB.rect(x, y, W-4*z, H-4*z, c4(0.28, 0.28, 0.3, 1))
	py := y + H*0.22
	for i := range 9 {
		lx := x - W*0.42 + float64(i)*W*0.105
		r.bushB.rect(lx, py, 1.2*z, H*0.3, c4(0.9, 0.9, 0.85, 0.8))
	}
	// A couple of parked cars
	carCols := [][3]float64{{0.75, 0.15, 0.15}, {0.2, 0.35, 0.7}, {0.85, 0.85, 0.8}}
	for i, c := range carCols {
		cx := x - W*0.37 + float64(i*3+1)*W*0.105
		r.shadowB.rect(cx+2*z, py+2*z, 7*z, 13*z, c4(0, 0, 0, 0.35))
		r.bushB.rect(cx, py, 7*z, 13*z, c4(c[0], c[1], c[2], 1))
		r.bushB.rect(cx, py-2*z, 5.5*z, 5*z, c4(0.15, 0.2, 0.25, 0.9))
	}
	// Building
	bx, by, bw, bh := x, y-H*0.2, W*0.86, H*0.52
	r.shadowB.rect(bx+shx*6*z, by+shy*6*z, bw, bh, c4(0, 0, 0, 0.4))
	r.bushB.rect(bx, by, bw, bh, c4(0.82, 0.82, 0.79, 1))
	r.bushB.rect(bx, by+bh*0.4, bw, bh*0.2, c4(0.1, 0.33, 0.72, 1))
	for i := range 3 {
		r.bushB.rect(bx-bw*0.3+float64(i)*bw*0.3, by-bh*0.2, 9*z, 7*z, c4(0.6, 0.62, 0.62, 1))
	}
	r.bushB.rect(bx, by+bh*0.48, 16*z, 4*z, c4(0.08, 0.1, 0.12, 1))
	// Yellow spark logo, glowing at night
	lx, ly := bx+bw*0.33, by+bh*0.4
	r.fxB.Quad(SprStar, lx, ly, 9*z, 9*z, 0, 1, 0.8, 0.1, 0.35+0.6*night)
	// Parking lamps
	for _, cx := range []float64{-0.45, 0, 0.45} {
		px, pyy := x+cx*W, y+H*0.46
		r.bushB.Circle(px, pyy, 1.6*z, c4(0.15, 0.15, 0.15, 1))
		r.fxB.Glow(px, pyy, 5*z, 1, 0.9, 0.7, 0.2+0.7*night)
		r.lightB.Glow(px, pyy, 90*z, 1, 0.85, 0.6, 0.9*night)
	}
	r.lightB.Glow(bx, by, 160*z, 0.9, 0.95, 1, 0.5*night)
}

func (r *Renderer) sprayCloud(p sim.Vec2, dir float64) {
	for range 26 {
		a := dir + rnd(-sim.SprayCone, sim.SprayCone)
		sp := rnd(1.5, 4)
		r.parts.Add(Particle{X: p.X, Y: p.Y, Z: 3, VX: math.Cos(a) * sp, VY: math.Sin(a) * sp, Drag: 0.06, Max: rnd(50, 90),
			Size: rnd(4, 9), Grow: 0.25, R: 0.85, G: 0.95, B: 0.8, A: 0.35, Spr: SprSmoke, Rot: rnd(0, 6)})
	}
	for range 12 {
		a := dir + rnd(-sim.SprayCone, sim.SprayCone)
		sp := rnd(3, 6)
		r.parts.Add(Particle{X: p.X, Y: p.Y, Z: 3, VX: math.Cos(a) * sp, VY: math.Sin(a) * sp, Grav: 0.08, Drag: 0.03, Max: 40,
			Size: rnd(0.6, 1.2), R: 0.8, G: 1, B: 0.7, A: 0.9, Spr: SprDot})
	}
}

func (r *Renderer) shopBuy(p sim.Vec2) {
	for range 10 {
		q := p.Add(sim.V(rnd(-20, 20), rnd(-10, 10)))
		r.parts.Add(Particle{X: q.X, Y: q.Y, VZ: rnd(0.4, 1), Max: 50, Size: rnd(1.5, 3), R: 1, G: 0.85, B: 0.3, A: 1, Spr: SprStar, Add: true, Rot: rnd(0, 1)})
	}
	r.parts.Add(Particle{X: p.X, Y: p.Y, Max: 30, Size: 10, Grow: 1.5, R: 1, G: 0.85, B: 0.3, A: 0.5, Spr: SprRing, Add: true, Light: 0.4})
}
