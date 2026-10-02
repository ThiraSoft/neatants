package render

import (
	"github.com/ThiraSoft/neatants/internal/sim"
	"math"
	"math/rand"

	"github.com/hajimehoshi/ebiten/v2/text/v2"
)

// Rendering of the Wallmart extras, the guard, couriers, frogs, confetti and
// the queens' live streams.

func (r *Renderer) drawWTF(w *sim.World, shx, shy float64) {
	r.drawSaleNeon(w)
	r.drawKiosk(w, shx, shy)
	r.drawGuard(w, shx, shy)
	r.drawCouriers(w, shx, shy)
	r.drawFrogs(w, shx, shy)
	r.drawLives(w)
	r.drawWeddings(w)
	r.drawEggs(w)
}

// drawEggs piles the colony's brood around the nest entrance.
func (r *Renderer) drawEggs(w *sim.World) {
	z := r.cam.Z
	if z < 0.5 {
		return
	}
	for _, c := range w.Colonies {
		if !c.Alive || c.Eggs == 0 || !r.visible(c.Pos, 60) {
			continue
		}
		x, y := r.ws(c.Pos)
		rng := rand.New(rand.NewSource(int64(c.ID*977 + 3)))
		for i := range c.Eggs {
			a := float64(i)*2.4 + rng.Float64()*0.4
			d := (sim.NestRadius*0.42 + float64(i%3)*4) * z
			ex, ey := x+math.Cos(a)*d, y+math.Sin(a)*d*0.8
			r.shadowB.Circle(ex+0.8*z, ey+0.8*z, 2*z, c4(0, 0, 0, 0.3))
			r.topB.Ellipse(ex, ey, 1.9*z, 1.3*z, a, c4(0.97, 0.95, 0.86, 1))
			r.topB.Circle(ex-0.5*z, ey-0.4*z, 0.6*z, c4(1, 1, 1, 0.8))
		}
	}
}

func (r *Renderer) drawSaleNeon(w *sim.World) {
	if !w.OnSale() || !r.visible(w.Shop, sim.ShopW*2) {
		return
	}
	z := r.cam.Z
	x, y := r.ws(w.Shop)
	blink := math.Sin(r.time*8) > 0
	cr, cg, cb := float32(1), float32(0.2), float32(0.25)
	if blink {
		cr, cg, cb = 1, 0.85, 0.15
	}
	r.fxB.Quad(SprRing, x, y-sim.ShopH*0.2*z, sim.ShopW*0.55*z, sim.ShopH*0.4*z, 0, cr, cg, cb, 0.45)
	r.lightB.Glow(x, y, 260*z, cr, cg, cb, 0.6)
	if rand.Float64() < 0.5 {
		p := w.Shop.Add(sim.V(rnd(-sim.ShopW, sim.ShopW), rnd(-sim.ShopH, sim.ShopH)))
		r.confetto(p, 30)
	}
}

func (r *Renderer) confetto(p sim.Vec2, z0 float64) {
	h := rand.Float64()
	c := hsv(h, 0.75, 1)
	r.parts.Add(Particle{X: p.X, Y: p.Y, Z: z0, VX: rnd(-0.6, 0.6), VY: rnd(-0.3, 0.3), VZ: rnd(-0.6, -0.2), Max: rnd(120, 220),
		Size: rnd(1.2, 2.2), Rot: rnd(0, 6), VRot: rnd(-0.3, 0.3), R: float32(c[0]), G: float32(c[1]), B: float32(c[2]), A: 1, Spr: SprSolid})
}

func hsv(h, s, v float64) [3]float64 {
	i := math.Floor(h * 6)
	f := h*6 - i
	p, q, t := v*(1-s), v*(1-f*s), v*(1-(1-f)*s)
	switch int(i) % 6 {
	case 0:
		return [3]float64{v, t, p}
	case 1:
		return [3]float64{q, v, p}
	case 2:
		return [3]float64{p, v, t}
	case 3:
		return [3]float64{p, q, v}
	case 4:
		return [3]float64{t, p, v}
	}
	return [3]float64{v, p, q}
}

func (r *Renderer) drawKiosk(w *sim.World, shx, shy float64) {
	k := w.Kiosk
	if k == nil || !k.Alive || !r.visible(k.Pos, 60) {
		return
	}
	z := r.cam.Z
	x, y := r.ws(k.Pos)
	col := w.Colonies[k.Owner].Color
	r.shadowB.rect(x+shx*4*z, y+shy*4*z, 44*z, 30*z, c4(0, 0, 0, 0.35))
	r.bushB.rect(x, y+4*z, 40*z, 20*z, c4(0.5, 0.34, 0.18, 1))
	for i := range 6 {
		c := [3]float64{0.95, 0.95, 0.92}
		if i%2 == 0 {
			c = col
		}
		r.bushB.rect(x-17.5*z+float64(i)*7*z, y-8*z, 7*z, 12*z, c4(c[0], c[1], c[2], 1))
	}
	for i := range min(k.Stock, 8) {
		cx := x - 14*z + float64(i%4)*9*z
		cy := y + 2*z + float64(i/4)*7*z
		r.bushB.rect(cx, cy, 5*z, 6*z, c4(0.85, 0.15, 0.12, 1))
	}
	r.lightB.Glow(x, y, 80*z, 1, 0.85, 0.6, float32(0.6*r.night))
}

func (r *Renderer) drawGuard(w *sim.World, shx, shy float64) {
	g := w.Guard
	if g == nil || !sim.Cfg.ShopEnabled || !r.visible(g.Pos, 40) {
		return
	}
	z := r.cam.Z
	x, y := r.ws(g.Pos)
	s := z * 1.3
	h := g.Heading
	fx, fy := math.Cos(h), math.Sin(h)
	px, py := -fy, fx
	at := func(f, p float64) (float64, float64) { return x + fx*f*s + px*p*s, y + fy*f*s + py*p*s }
	r.shadowB.Quad(SprGlow, x+shx*3*s, y+shy*3*s, 14*s, 10*s, h, 0, 0, 0, 0.4)
	lc := c4(0.08, 0.1, 0.18, 1)
	for i := -1; i <= 1; i++ {
		for _, side := range []float64{-1, 1} {
			ph := math.Sin(g.Gait*1.2+float64(i)*math.Pi+side) * 1.5
			ax, ay := at(float64(i)*2.5, side*4)
			tx, ty := at(float64(i)*5+ph, side*10)
			r.sceneB.Line(SprDot, ax, ay, tx, ty, 1.6*s, lc)
		}
	}
	bx, by := at(-2, 0)
	r.sceneB.Ellipse(bx, by, 9*s, 7*s, h, c4(0.12, 0.16, 0.3, 1))
	vx, vy := at(-1, 0)
	r.sceneB.Quad(SprSolid, vx, vy, 1.6*s, 6.8*s, h, 1, 0.85, 0.1, 1) // hi-vis vest band
	hx, hy := at(7, 0)
	r.sceneB.Circle(hx, hy, 3.6*s, c4(0.08, 0.1, 0.2, 1))
	cx, cy := at(8.5, 0)
	r.sceneB.Ellipse(cx, cy, 2.2*s, 3.4*s, h, c4(0.05, 0.07, 0.15, 1)) // cap visor
	if g.Target != nil {
		red := math.Sin(r.time*14) > 0
		cr, cb := float32(1), float32(0.15)
		if !red {
			cr, cb = 0.15, 1
		}
		r.fxB.Glow(hx, hy, 9*s, cr, 0.15, cb, 0.9)
		r.lightB.Glow(hx, hy, 70*s, cr, 0.15, cb, 0.7)
	}
	if r.night > 0.2 {
		// Flashlight beam
		lx, ly := at(30, 0)
		r.fxB.Quad(SprGlow, lx, ly, 26*s, 12*s, h, 1, 0.95, 0.8, float32(0.18*r.night))
		r.lightB.Quad(SprGlow, lx, ly, 40*s, 22*s, h, 1, 0.95, 0.85, float32(0.8*r.night))
	}
}

func (r *Renderer) drawCouriers(w *sim.World, shx, shy float64) {
	z := r.cam.Z
	for _, k := range w.Couriers {
		if !k.Alive || !r.visible(k.Pos, 40) {
			continue
		}
		x, y := r.ws(k.Pos)
		s := z * 1.2
		h := k.Angle
		fx, fy := math.Cos(h), math.Sin(h)
		at := func(f float64) (float64, float64) { return x + fx*f*s, y + fy*f*s }
		r.shadowB.Quad(SprGlow, x+shx*2*s, y+shy*2*s, 12*s, 5*s, h, 0, 0, 0, 0.35)
		r.sceneB.Quad(SprSolid, x, y, 9*s, 2*s, h, 0.3, 0.32, 0.35, 1)
		for _, f := range []float64{-8, 8} {
			wx, wy := at(f)
			r.sceneB.Circle(wx, wy, 2.4*s, c4(0.05, 0.05, 0.05, 1))
		}
		sx, sy := at(7)
		r.sceneB.Circle(sx, sy, 1.4*s, c4(0.6, 0.6, 0.62, 1))
		bx, by := at(-1)
		r.sceneB.Ellipse(bx, by, 4.5*s, 3.2*s, h, c4(0.35, 0.6, 0.2, 1)) // cricket rider
		hx, hy := at(2.5)
		r.sceneB.Circle(hx, hy, 2.6*s, c4(0.85, 0.15, 0.12, 1)) // helmet
		r.sceneB.Quad(SprSolid, bx-fx*4*s, by-fy*4*s, 2.2*s, 2.2*s, h, 0.95, 0.95, 0.92, 1)
		for i := range 3 {
			off := float64(i-1) * 3 * s
			lx, ly := x-fy*off-fx*12*s, y+fx*off-fy*12*s
			r.fxB.Line(SprGlow, lx, ly, lx-fx*14*s, ly-fy*14*s, 1.5*s, c4(1, 1, 1, 0.25))
		}
		fx0, fy0 := at(12)
		r.fxB.Glow(fx0, fy0, 3*s, 1, 0.95, 0.7, float32(0.3+0.6*r.night))
		lx, ly := at(40)
		r.lightB.Quad(SprGlow, lx, ly, 34*s, 14*s, h, 1, 0.95, 0.8, float32(0.8*r.night))
	}
}

func (r *Renderer) drawFrogs(w *sim.World, shx, shy float64) {
	z := r.cam.Z
	for _, f := range w.Frogs {
		if !r.visible(f.Pos, 40) {
			continue
		}
		x, y := r.ws(f.Pos)
		arc := math.Sin(f.HopT*math.Pi) * 10
		lift := (f.Z + arc) * z
		s := z * 1.1
		fade := math.Min(1, float64(f.Life)/60)
		r.shadowB.Circle(x+shx*s, y+shy*s, (9-math.Min(6, f.Z*0.02))*s, c4(0, 0, 0, 0.35*fade))
		y -= lift
		body := c4(0.3, 0.62, 0.22, fade)
		if f.HopT > 0 || f.Z > 0 { // legs stretched mid-air
			for _, side := range []float64{-1, 1} {
				r.sceneB.Ellipse(x+side*6*s, y+6*s, 2.5*s, 6*s, side*0.5, c4(0.25, 0.52, 0.18, fade))
			}
		} else {
			for _, side := range []float64{-1, 1} {
				r.sceneB.Ellipse(x+side*7*s, y+3*s, 3.5*s, 4.5*s, 0, c4(0.25, 0.52, 0.18, fade))
			}
		}
		r.sceneB.Ellipse(x, y, 8*s, 7*s, 0, body)
		r.sceneB.Ellipse(x, y+2*s, 5*s, 4*s, 0, c4(0.75, 0.85, 0.5, 0.7*fade))
		for _, side := range []float64{-1, 1} {
			ex, ey := x+side*4*s, y-5*s
			r.sceneB.Circle(ex, ey, 2.8*s, body)
			r.sceneB.Circle(ex, ey, 1.9*s, c4(0.95, 0.95, 0.85, fade))
			r.sceneB.Circle(ex, ey, 1*s, c4(0.05, 0.05, 0.05, fade))
		}
		if f.Tongue > 0.1 {
			lx, ly := r.ws(f.Lick)
			tx, ty := x+(lx-x)*f.Tongue, y+(ly-y)*f.Tongue
			r.topB.Line(SprDot, x, y+2*s, tx, ty, 1.6*s, c4(0.95, 0.4, 0.5, 1))
			r.topB.Circle(tx, ty, 1.8*s, c4(0.95, 0.4, 0.5, 1))
		}
	}
}

func (r *Renderer) drawLives(w *sim.World) {
	z := r.cam.Z
	for _, c := range w.Colonies {
		if !c.Alive || !c.Streaming(w.Tick) || !r.visible(c.Pos, 200) {
			continue
		}
		x, y := r.ws(c.Pos)
		pulse := float32(0.85 + 0.15*math.Sin(r.time*3))
		// The ring light
		r.fxB.Quad(SprRing, x, y-4*z, 26*z, 26*z, 0, 1, 1, 1, 0.6*pulse)
		r.fxB.Glow(x, y, 40*z, 1, 0.95, 0.9, 0.15)
		r.lightB.Glow(x, y, 200*z, 1, 0.95, 0.95, float32(0.2+0.25*r.day))
		if rand.Float64() < 0.5 {
			p := c.Pos.Add(sim.Polar(rnd(0, 6.28), rnd(20, 150)))
			col := [3]float32{1, 0.35, 0.55}
			if rand.Float64() < 0.3 {
				col = [3]float32{float32(c.Color[0]), float32(c.Color[1]), float32(c.Color[2])}
			}
			r.parts.Add(Particle{X: p.X, Y: p.Y, Z: 4, VZ: rnd(0.3, 0.7), VX: rnd(-0.15, 0.15), Max: rnd(70, 120), Size: rnd(2, 3.5),
				R: col[0], G: col[1], B: col[2], A: 1, Spr: SprHeart, Add: true, Light: 0.1})
		}
	}
}

// Weird weather ambience (confetti storm) and event visuals.
func (r *Renderer) weirdAmbient(w *sim.World) {
	if w.Weird == sim.WeirdConfetti {
		for range 10 {
			r.confetto(r.sw(rnd(-50, SW+50), rnd(-50, SH+50)), rnd(40, 120))
		}
	}
}

func (r *Renderer) onWTFEvent(w *sim.World, e sim.Event, vis bool) {
	p := e.Pos
	r.onWeddingEvent(w, e)
	switch e.Kind {
	case sim.EvSaleStart:
		r.banner = Banner{Title: "Black Friday", Sub: "Insecticide at -70 % at Wallmart", T: 5, Col: [3]float64{1, 0.8, 0.2}}
		r.note(p, 9)
		for range 60 {
			r.confetto(p.Add(sim.V(rnd(-120, 120), rnd(-80, 80))), rnd(20, 80))
		}
	case sim.EvStolen:
		if vis {
			r.parts.Add(Particle{X: p.X, Y: p.Y, Z: 10, VZ: 0.4, Max: 40, Size: 6, R: 1, G: 0.25, B: 0.2, A: 1, Spr: SprStar, Add: true, Light: 0.5})
		}
		r.note(p, 3)
	case sim.EvGuardBite, sim.EvCourierHit:
		if vis {
			r.parts.Add(Particle{X: p.X, Y: p.Y, Max: 12, Size: 5, Grow: 0.5, R: 1, G: 0.95, B: 0.7, A: 0.9, Spr: SprStar, Add: true, Rot: rnd(0, 3)})
			for range 5 {
				r.parts.Add(Particle{X: p.X, Y: p.Y, VX: rnd(-1.5, 1.5), VY: rnd(-1.5, 1.5), Drag: 0.08, Max: 30, Size: rnd(3, 6), Grow: 0.15,
					R: 0.55, G: 0.5, B: 0.42, A: 0.4, Spr: SprSmoke})
			}
		}
	case sim.EvKioskOpen:
		r.shopBuy(p)
		r.note(p, 5)
	case sim.EvCourierArrive:
		if vis {
			r.parts.Add(Particle{X: p.X, Y: p.Y, Max: 30, Size: 12, Grow: 2, R: 1, G: 0.9, B: 0.5, A: 0.6, Spr: SprRing, Add: true, Light: 0.4})
		}
	case sim.EvWeirdStart:
		if int(e.Val) == sim.WeirdFrogs {
			r.banner = Banner{Title: "Frog rain", Sub: "They eat everything that moves", T: 4.5, Col: [3]float64{0.5, 0.95, 0.4}}
		} else {
			r.banner = Banner{Title: "Confetti storm", Sub: "Every trail is scrambled", T: 4.5, Col: [3]float64{1, 0.55, 0.9}}
		}
	case sim.EvFrogLand:
		if vis {
			r.parts.Add(Particle{X: p.X, Y: p.Y, Max: 20, Size: 6, Grow: 1.2, R: 0.85, G: 0.9, B: 0.8, A: 0.5, Spr: SprRing})
			for range 6 {
				a := rnd(0, 6.28)
				r.parts.Add(Particle{X: p.X, Y: p.Y, VX: math.Cos(a) * 1.5, VY: math.Sin(a) * 1.5, Drag: 0.06, Max: 30, Size: rnd(3, 6), Grow: 0.15,
					R: 0.5, G: 0.45, B: 0.35, A: 0.4, Spr: SprSmoke})
			}
		}
	case sim.EvFrogEat:
		r.note(p, 2)
	case sim.EvLiveStart:
		r.note(p, 6)
	case sim.EvNestMove:
		old := e.Pts[0]
		for range 40 {
			q := old.Add(sim.Polar(rnd(0, 6.28), rnd(0, 60)))
			r.parts.Add(Particle{X: q.X, Y: q.Y, VX: rnd(-0.6, 0.6), VY: rnd(-0.6, 0.6), VZ: 0.2, Drag: 0.02, Max: rnd(90, 160),
				Size: rnd(12, 26), Grow: 0.25, R: 0.4, G: 0.33, B: 0.26, A: 0.45, Spr: SprSmoke, Rot: rnd(0, 6)})
		}
		r.decal(SprCrack, old, 140, rnd(0, 6), [4]float32{1, 1, 1, 0.8})
		r.shopBuy(p)
		r.note(p, 6)
	}
}

// drawWTFLabels draws the floating texts of the extras.
func (u *UI) drawWTFLabels(g *Game, draw func(s string, face *text.GoTextFace, x, y float64, c [3]float64, a float64)) {
	w, r := g.World, g.R
	z := r.cam.Z
	if z < 0.22 {
		return
	}
	if sim.Cfg.ShopEnabled && r.visible(w.Shop, sim.ShopW) {
		x, y := r.ws(w.Shop)
		if w.OnSale() {
			c := [3]float64{1, 0.85, 0.2}
			if math.Sin(r.time*8) > 0 {
				c = [3]float64{1, 0.3, 0.3}
			}
			draw("BLACK FRIDAY -70 %", u.f.Title, x, y-sim.ShopH*0.46*z-56, c, 1)
		}
		draw(itoa(w.ShopStock)+" packs on the shelf", u.f.Tiny, x, y+sim.ShopH*0.5*z+4, colMuted, 0.9)
	}
	if k := w.Kiosk; k != nil && k.Alive && r.visible(k.Pos, 60) {
		x, y := r.ws(k.Pos)
		c := w.Colonies[k.Owner]
		draw("Coop "+c.Name, u.f.Small, x, y-22*z-16, c.Color, 0.95)
		draw(itoa(int(math.Round(k.Price)))+" · stock "+itoa(k.Stock), u.f.Tiny, x, y+14*z+2, colText, 0.85)
	}
	if gd := w.Guard; gd != nil && sim.Cfg.ShopEnabled && gd.Target != nil && r.visible(gd.Pos, 30) {
		x, y := r.ws(gd.Pos)
		draw("Stop, thief!", u.f.Small, x, y-20*z-14, [3]float64{1, 0.4, 0.3}, 1)
	}
	for _, c := range w.Colonies {
		if c.Alive && c.Streaming(w.Tick) && r.visible(c.Pos, 100) {
			x, y := r.ws(c.Pos)
			draw("● LIVE · "+sim.FmtLikes(c.Likes)+" likes", u.f.H, x, y-sim.NestRadius*2.3*z-48, [3]float64{1, 0.42, 0.45}, 1)
		}
	}
}

// --- Marriages ---

func (r *Renderer) drawWeddings(w *sim.World) {
	z := r.cam.Z
	for _, a := range w.Colonies {
		s := w.SpouseOf(a.ID)
		if s < 0 || a.ID > s {
			continue
		}
		arch := w.WeddingArch[a.ID]
		if !r.visible(arch, 80) {
			continue
		}
		x, y := r.ws(arch)
		k := 2.0 * z // the arch is drawn twice life size so it reads from afar
		// Two posts and a half ring of flowers.
		for _, side := range []float64{-1, 1} {
			r.shadowB.Circle(x+side*22*k+2*k, y+3*k, 3*k, c4(0, 0, 0, 0.35))
			r.bushB.rect(x+side*22*k, y-8*k, 3*k, 18*k, c4(0.45, 0.32, 0.2, 1))
		}
		flowers := [][3]float64{{1, 0.55, 0.7}, {1, 1, 0.95}, {1, 0.85, 0.35}}
		for i := range 15 {
			ang := math.Pi + float64(i)/14*math.Pi
			fx, fy := x+math.Cos(ang)*22*k, y-8*k+math.Sin(ang)*20*k
			r.bushB.Circle(fx, fy, 3.4*k, c4(0.25, 0.5, 0.2, 1))
			c := flowers[i%3]
			r.bushB.Circle(fx, fy-1*k, 2.2*k, c4(c[0], c[1], c[2], 1))
		}
		r.fxB.Quad(SprHeart, x, y-10*k, 7*k, 7*k, 0, 1, 0.4, 0.6, float32(0.4+0.4*r.night))
		r.lightB.Glow(x, y, 90*k, 1, 0.6, 0.75, float32(0.5*r.night))
		if rand.Float64() < 0.08 {
			p := arch.Add(sim.V(rnd(-20, 20), rnd(-10, 5)))
			r.parts.Add(Particle{X: p.X, Y: p.Y, Z: 18, VZ: 0.3, VX: rnd(-0.1, 0.1), Max: 90, Size: rnd(1.5, 2.5),
				R: 1, G: 0.45, B: 0.65, A: 1, Spr: SprHeart, Add: true})
		}
	}
}

func (r *Renderer) petals(p sim.Vec2, n int) {
	for range n {
		q := p.Add(sim.V(rnd(-60, 60), rnd(-40, 40)))
		c := [3]float32{1, 0.7, 0.8}
		if rand.Float64() < 0.4 {
			c = [3]float32{1, 1, 0.95}
		}
		r.parts.Add(Particle{X: q.X, Y: q.Y, Z: rnd(30, 90), VX: rnd(-0.5, 0.5), VY: rnd(-0.3, 0.3), Grav: 0.02, Drag: 0.01, Max: rnd(120, 200),
			Size: rnd(2, 3.5), Rot: rnd(0, 6), VRot: rnd(-0.1, 0.1), R: c[0], G: c[1], B: c[2], A: 0.95, Spr: SprLeaf})
	}
}

func (r *Renderer) heartBurst(p sim.Vec2, n int) {
	for range n {
		a := rnd(0, 6.28)
		r.parts.Add(Particle{X: p.X, Y: p.Y, Z: 6, VX: math.Cos(a) * rnd(0.5, 2), VY: math.Sin(a) * rnd(0.5, 2), VZ: rnd(0.3, 1), Drag: 0.03,
			Max: rnd(60, 110), Size: rnd(2, 4), R: 1, G: 0.4, B: 0.6, A: 1, Spr: SprHeart, Add: true, Light: 0.15})
	}
}

func (r *Renderer) onWeddingEvent(w *sim.World, e sim.Event) {
	switch e.Kind {
	case sim.EvWedding:
		a, b := w.Colonies[e.Colony], w.Colonies[int(e.Val)]
		r.banner = Banner{Title: "Wedding!", Sub: a.Name + " and " + b.Name + " join their colonies", T: 5, Col: [3]float64{1, 0.55, 0.75}}
		r.petals(e.Pos, 70)
		r.heartBurst(e.Pos, 30)
		for _, p := range e.Pts {
			r.heartBurst(p, 20)
		}
		r.note(e.Pos, 9)
	case sim.EvDivorce:
		a, b := w.Colonies[e.Colony], w.Colonies[int(e.Val)]
		r.banner = Banner{Title: "Divorce", Sub: a.Name + " and " + b.Name + " split up", T: 4.5, Col: [3]float64{0.7, 0.65, 0.8}}
		for _, side := range []float64{-1, 1} {
			r.parts.Add(Particle{X: e.Pos.X, Y: e.Pos.Y, Z: 20, VX: side * 0.8, VZ: 0.2, Rot: side * 0.6, VRot: side * 0.02, Max: 120, Size: 14,
				R: 0.9, G: 0.3, B: 0.45, A: 1, Spr: SprHeart, Add: true, Light: 0.3})
		}
		for range 12 {
			r.parts.Add(Particle{X: e.Pos.X + rnd(-20, 20), Y: e.Pos.Y + rnd(-20, 20), VZ: 0.3, Max: 100, Size: rnd(10, 18), Grow: 0.2,
				R: 0.4, G: 0.38, B: 0.42, A: 0.4, Spr: SprSmoke, Rot: rnd(0, 6)})
		}
		r.note(e.Pos, 6)
	case sim.EvGift:
		if !r.visible(e.Pos, 400) && !r.visible(e.Pts[0], 400) {
			return
		}
		from, to := e.Pts[0], e.Pos
		d := to.Sub(from)
		for k := range 5 {
			q := from.Add(d.Scale(float64(k) * 0.02))
			r.parts.Add(Particle{X: q.X, Y: q.Y, Z: 25, VX: d.X / 110, VY: d.Y / 110, Max: 105, Size: 2.5,
				R: 1, G: 0.8, B: 0.4, A: 0.9, Spr: SprHeart, Add: true, Light: 0.1})
		}
	}
}
