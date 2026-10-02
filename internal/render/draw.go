package render

import (
	"github.com/ThiraSoft/neatants/internal/sim"
	"math"
	"math/rand"

	"github.com/hajimehoshi/ebiten/v2"
)

type col4 = [4]float32

func c4(r, g, b, a float64) col4 { return col4{float32(r), float32(g), float32(b), float32(a)} }

func (r *Renderer) Draw(w *sim.World) *ebiten.Image {
	z := r.cam.Z
	sx0, sy0 := 0.0, 0.0
	if r.shake > 0.1 {
		sx0, sy0 = rnd(-r.shake, r.shake), rnd(-r.shake, r.shake)
	}
	camX, camY := r.cam.X, r.cam.Y
	r.cam.X -= sx0 / z
	r.cam.Y -= sy0 / z
	defer func() { r.cam.X, r.cam.Y = camX, camY }()

	// 1. Terrain
	r.drawTerrain(w)

	// 2. Ground layers: decals, worn paths, pheromones
	r.decalB.Flush(r.decals, ebiten.BlendSourceOver)
	if r.frame%8 == 0 {
		fade(r.decals)
	}
	for _, a := range w.Ants {
		if a.Alive {
			r.pathB.Quad(SprSoft, a.Pos.X/4, a.Pos.Y/4, 1.3, 1.3, 0, 0.12, 0.08, 0.05, 0.05)
		}
	}
	r.pathB.Flush(r.paths, ebiten.BlendSourceOver)
	if r.frame%5 == 0 {
		fade(r.paths)
	}
	r.drawLayer(r.paths, 4, 0.55, ebiten.BlendSourceOver, r.scene)
	r.drawLayer(r.decals, 2, 1, ebiten.BlendSourceOver, r.scene)
	if r.ShowPhero {
		r.updatePhero(w)
		r.drawLayer(r.phero, sim.Cell, 0.4, ebiten.BlendSourceOver, r.scene)
	}

	// 3. World objects
	r.light.Clear()
	r.fx.Clear()
	if r.ShowPhero && r.night > 0.05 {
		r.drawLayer(r.phero, sim.Cell, float32(r.night*0.2), ebiten.BlendLighter, r.fx)
	}
	shOx, shOy, shA := r.sunShadow()
	_ = shA
	r.drawNests(w)
	r.drawShop(w, shOx, shOy)
	r.drawWTF(w, shOx, shOy)
	r.drawBushes(w, shOx, shOy)
	r.drawFood(w)
	for _, a := range w.Ants {
		if a.Alive {
			r.drawAnt(w, a, shOx, shOy)
		}
	}
	for _, m := range w.Monsters {
		if m.Alive {
			r.drawMonster(w, m, shOx, shOy)
		}
	}
	r.drawFireballs(w)
	r.drawBolts()
	r.drawParticles()
	r.drawRain()

	r.shadowB.Flush(r.scene, ebiten.BlendSourceOver)
	r.bushB.Flush(r.scene, ebiten.BlendSourceOver)
	r.sceneB.Flush(r.scene, ebiten.BlendSourceOver)
	r.topB.Flush(r.scene, ebiten.BlendSourceOver)
	r.fxB.Flush(r.fx, ebiten.BlendLighter)
	r.lightB.Flush(r.light, ebiten.BlendLighter)

	// 4. Bloom from emissive buffer
	r.bloomA.Clear()
	op := &ebiten.DrawImageOptions{Filter: ebiten.FilterLinear}
	op.GeoM.Scale(0.25, 0.25)
	r.bloomA.DrawImage(r.fx, op)
	r.blur(r.bloomA, r.bloomB, 1)
	r.blur(r.bloomA, r.bloomB, 2.2)
	r.bloomUp.Clear()
	op = &ebiten.DrawImageOptions{Filter: ebiten.FilterLinear}
	op.GeoM.Scale(4, 4)
	r.bloomUp.DrawImage(r.bloomA, op)

	// 5. Composite
	r.composite(w)
	return r.final
}

func fade(img *ebiten.Image) {
	op := &ebiten.DrawImageOptions{Blend: blendSubtract}
	b := img.Bounds()
	op.GeoM.Scale(float64(b.Dx()), float64(b.Dy()))
	op.ColorScale.Scale(1.0/255, 1.0/255, 1.0/255, 1.0/255)
	img.DrawImage(whiteSub, op)
}

// drawLayer draws a world-aligned image whose pixels cover `scale` world units.
func (r *Renderer) drawLayer(img *ebiten.Image, scale float64, alpha float32, blend ebiten.Blend, dst *ebiten.Image) {
	op := &ebiten.DrawImageOptions{Filter: ebiten.FilterLinear, Blend: blend}
	op.GeoM.Scale(scale*r.cam.Z, scale*r.cam.Z)
	op.GeoM.Translate(-r.cam.X*r.cam.Z+SW/2, -r.cam.Y*r.cam.Z+SH/2)
	op.ColorScale.ScaleAlpha(alpha)
	dst.DrawImage(img, op)
}

func (r *Renderer) blur(a, tmp *ebiten.Image, spread float32) {
	b := a.Bounds()
	op := &ebiten.DrawRectShaderOptions{}
	op.Images[0] = a
	op.Uniforms = map[string]any{"Dir": []float32{spread, 0}}
	tmp.Clear()
	tmp.DrawRectShader(b.Dx(), b.Dy(), r.blurSh, op)
	op = &ebiten.DrawRectShaderOptions{}
	op.Images[0] = tmp
	op.Uniforms = map[string]any{"Dir": []float32{0, spread}}
	a.Clear()
	a.DrawRectShader(b.Dx(), b.Dy(), r.blurSh, op)
}

func (r *Renderer) drawTerrain(w *sim.World) {
	ponds := make([]float32, 8*4)
	for i, p := range w.Ponds {
		if i >= 8 {
			break
		}
		ponds[i*4], ponds[i*4+1], ponds[i*4+2] = float32(p.Pos.X), float32(p.Pos.Y), float32(p.R)
	}
	nests := make([]float32, 6*4)
	ncol := make([]float32, 6*4)
	for i, c := range w.Colonies {
		if i >= 6 {
			break
		}
		nests[i*4], nests[i*4+1], nests[i*4+2] = float32(c.Pos.X), float32(c.Pos.Y), sim.NestRadius
		if c.Alive {
			nests[i*4+3] = 1
		}
		ncol[i*4], ncol[i*4+1], ncol[i*4+2], ncol[i*4+3] = float32(c.Color[0]), float32(c.Color[1]), float32(c.Color[2]), float32(c.HitFlash)
	}
	op := &ebiten.DrawRectShaderOptions{}
	op.Uniforms = map[string]any{
		"Cam":     []float32{float32(r.cam.X - SW/2/r.cam.Z), float32(r.cam.Y - SH/2/r.cam.Z)},
		"Zoom":    float32(r.cam.Z),
		"Time":    float32(r.time),
		"Wet":     float32(r.wet),
		"Cave":    []float32{float32(w.Cave.X), float32(w.Cave.Y)},
		"Ponds":   ponds,
		"Nests":   nests,
		"NestCol": ncol,
	}
	r.scene.DrawRectShader(SW, SH, r.terrainSh, op)
}

func (r *Renderer) composite(w *sim.World) {
	day := [3]float64{1.0, 0.98, 0.94}
	dusk := [3]float64{1.0, 0.66, 0.48}
	night := [3]float64{0.27, 0.32, 0.5}
	amb := sim.Mixc(night, day, r.day)
	duskW := math.Exp(-r.elev * r.elev * 18)
	amb = sim.Mixc(amb, [3]float64{amb[0] * dusk[0] * 1.05, amb[1] * dusk[1], amb[2] * dusk[2]}, duskW*0.8)
	amb = sim.Mixc(amb, [3]float64{amb[0] * 0.72, amb[1] * 0.76, amb[2] * 0.85}, r.wet)

	shocks := make([]float32, 16*4)
	for i, s := range r.shocks {
		x, y := r.ws(s.Pos)
		t := s.T / s.Max
		shocks[i*4] = float32(x)
		shocks[i*4+1] = float32(y)
		shocks[i*4+2] = float32((s.R0 + (s.R1-s.R0)*math.Sqrt(t)) * r.cam.Z)
		shocks[i*4+3] = float32(s.Str * (1 - t) * (1 - t) * math.Min(1, r.cam.Z*1.5))
	}
	op := &ebiten.DrawRectShaderOptions{}
	op.Images = [4]*ebiten.Image{r.scene, r.light, r.fx, r.bloomUp}
	op.Uniforms = map[string]any{
		"Ambient": []float32{float32(amb[0]), float32(amb[1]), float32(amb[2])},
		"LightK":  float32(0.3 + 1.5*r.night),
		"Time":    float32(r.time),
		"Shocks":  shocks,
		"Chroma":  float32(0.6 + r.shake*0.35),
		"Flash":   float32(r.flash),
		"Night":   float32(r.night),
		"Bloom":   float32(1.0 + 0.4*r.night),
	}
	r.final.DrawRectShader(SW, SH, r.compSh, op)
}

func (r *Renderer) sunShadow() (float64, float64, float64) {
	// Sun sweeps east to west; shadows lengthen near dusk.
	l := 1.2 + 2.5*(1-sim.ClampF(r.elev, 0, 1))
	a := math.Pi*0.25 + (r.elev)*0.4
	return math.Cos(a) * l, math.Sin(a) * l, 0.25 + 0.15*r.day
}

func (r *Renderer) updatePhero(w *sim.World) {
	pix := r.pheroPix
	for i := range sim.Cells {
		var cr, cg, cb, ca float64
		for k, c := range w.Colonies {
			if k >= sim.MaxColonies {
				break
			}
			t := float64(w.Trail[k][i])
			al := float64(w.Alarm[k][i])
			if t > 0.01 {
				cr += c.Color[0] * t
				cg += c.Color[1] * t
				cb += c.Color[2] * t
				ca += t
			}
			if al > 0.01 {
				al *= 0.5
				cr += 1 * al
				cg += 0.25 * al
				cb += 0.2 * al
				ca += al
			}
		}
		a := math.Min(ca, 1) * 0.5
		if ca > 0 {
			cr, cg, cb = cr/ca, cg/ca, cb/ca
		}
		pix[i*4] = byte(math.Min(cr, 1) * a * 255)
		pix[i*4+1] = byte(math.Min(cg, 1) * a * 255)
		pix[i*4+2] = byte(math.Min(cb, 1) * a * 255)
		pix[i*4+3] = byte(a * 255)
	}
	r.phero.WritePixels(pix)
}

// --- Entities ---

func (r *Renderer) drawNests(w *sim.World) {
	z := r.cam.Z
	for _, c := range w.Colonies {
		x, y := r.ws(c.Pos)
		if !r.visible(c.Pos, 300) {
			continue
		}
		cr, cg, cb := float32(c.Color[0]), float32(c.Color[1]), float32(c.Color[2])
		if !c.Alive {
			if rand.Float64() < 0.2 {
				p := c.Pos.Add(sim.Polar(rnd(0, 6.28), rnd(0, 50)))
				r.parts.Add(Particle{X: p.X, Y: p.Y, VZ: 0.4, Max: 120, Size: 8, Grow: 0.2, R: 0.3, G: 0.28, B: 0.27, A: 0.25, Spr: SprSmoke})
			}
			continue
		}
		intensity := float32(0.14 + 0.16*r.night + c.Pulse*0.35)
		rot := r.time * 0.12
		R := sim.NestRadius * 2.1 * z
		r.fxB.Quad(SprRune, x, y, R, R, rot, cr, cg, cb, intensity*0.55)
		r.fxB.Quad(SprRune, x, y, R*0.62, R*0.62, -rot*1.7, cr, cg, cb, intensity*0.35)
		r.fxB.Glow(x, y, 26*z, cr, cg, cb, intensity*0.3)
		r.fxB.Glow(x, y, 9*z, 1, 0.9, 0.7, intensity*0.6)
		r.lightB.Glow(x, y, 300*z, cr, cg, cb, float32(0.05+0.28*r.night+c.Pulse*0.3))
		r.lightB.Glow(x, y, 120*z, 1, 0.85, 0.6, float32(0.05+0.22*r.night))
		if c.HitFlash > 0.05 {
			r.fxB.Quad(SprRing, x, y, R*1.1, R*1.1, 0, 1, 0.2, 0.15, float32(c.HitFlash))
		}
		// Rising sparks of colony light
		if rand.Float64() < 0.15+0.3*r.night {
			p := c.Pos.Add(sim.Polar(rnd(0, 6.28), rnd(20, 70)))
			r.parts.Add(Particle{X: p.X, Y: p.Y, VZ: rnd(0.3, 0.8), Max: 90, Size: rnd(1, 2), R: cr, G: cg, B: cb, A: 1, Spr: SprGlow, Add: true, Light: 0.1})
		}
	}
	// Cave glow
	if r.visible(w.Cave, 600) {
		x, y := r.ws(w.Cave)
		pulse := float32(0.75 + 0.25*math.Sin(r.time*2.1) + 0.1*math.Sin(r.time*7.3))
		if w.WaveAlive > 0 {
			pulse *= 1.4
		}
		r.lightB.Glow(x, y, 520*z, 1, 0.3, 0.08, 0.8*pulse)
		r.fxB.Glow(x, y, 90*z, 1, 0.25, 0.05, 0.35*pulse)
		r.fxB.Quad(SprRune, x, y, 150*z, 150*z, -r.time*0.3, 1, 0.2, 0.1, 0.12*pulse+0.3*float32(sim.BoolF(len(w.SpawnQueue) > 0)))
		// Eyes in the dark
		for i := range 5 {
			ph := math.Sin(r.time*0.7 + float64(i)*2.1)
			if ph < 0.6 {
				continue
			}
			ex := w.Cave.X + math.Cos(float64(i)*1.7)*45
			ey := w.Cave.Y + math.Sin(float64(i)*2.3)*30
			sx, sy := r.ws(sim.V(ex, ey))
			a := float32((ph - 0.6) * 2.5)
			r.fxB.Glow(sx-3*z, sy, 3*z, 1, 0.15, 0.1, a)
			r.fxB.Glow(sx+3*z, sy, 3*z, 1, 0.15, 0.1, a)
		}
	}
}

func (r *Renderer) drawBushes(w *sim.World, shx, shy float64) {
	z := r.cam.Z
	for bi, b := range w.Bushes {
		if !r.visible(b.Pos, 120) {
			continue
		}
		x, y := r.ws(b.Pos)
		rng := rand.New(rand.NewSource(int64(bi*7919 + 13)))
		R := b.R * z
		r.shadowB.Quad(SprGlow, x+shx*5*z, y+shy*5*z, R*1.5, R*1.3, 0, 0, 0, 0, 0.45)
		for i := range 30 {
			a := rng.Float64() * 6.28
			d := math.Sqrt(rng.Float64()) * b.R * 0.9
			lr := (5 + rng.Float64()*8) * z
			sway := math.Sin(r.time*1.3+b.Seed+float64(i)*0.3) * 1.8 * (d / b.R) * z
			lx := x + math.Cos(a)*d*z + sway
			ly := y + math.Sin(a)*d*z + sway*0.4
			sh := 0.7 + 0.5*(1-d/b.R) + rng.Float64()*0.15
			g := [3]float64{0.2 * sh, 0.42 * sh, 0.14 * sh}
			if b.Kind == 3 {
				g = [3]float64{0.16 * sh, 0.36 * sh, 0.26 * sh}
			}
			r.bushB.Circle(lx, ly, lr, c4(g[0], g[1], g[2], 1))
			r.bushB.Circle(lx-lr*0.3, ly-lr*0.3, lr*0.45, c4(g[0]*1.35, g[1]*1.3, g[2]*1.2, 0.5))
		}
		fr := [3]float64{0.9, 0.2, 0.2}
		switch b.Kind {
		case 0:
			fr = [3]float64{1, 0.82, 0.35}
		case 3:
			fr = [3]float64{0.45, 0.55, 1}
		}
		for i := range 12 {
			a := rng.Float64() * 6.28
			d := rng.Float64() * b.R * 0.8
			fx, fy := x+math.Cos(a)*d*z, y+math.Sin(a)*d*z
			r.bushB.Circle(fx, fy, 2.2*z, c4(fr[0], fr[1], fr[2], 1))
			r.bushB.Circle(fx-0.6*z, fy-0.6*z, 0.8*z, c4(1, 1, 1, 0.6))
			if b.Kind == 3 {
				r.fxB.Glow(fx, fy, 6*z, 0.4, 0.55, 1, float32(0.15+0.5*r.night))
			}
			_ = i
		}
		if b.Kind == 3 {
			r.lightB.Glow(x, y, 110*z, 0.35, 0.5, 1, float32(0.35*r.night))
		}
	}
}

var foodColors = [4][3]float64{
	{0.92, 0.75, 0.34},
	{0.85, 0.16, 0.2},
	{0.62, 0.3, 0.38},
	{0.4, 0.5, 1},
}

func (r *Renderer) drawFood(w *sim.World) {
	z := r.cam.Z
	for i := range w.Foods {
		f := &w.Foods[i]
		if f.Taken || !r.visible(f.Pos, 10) {
			continue
		}
		x, y := r.ws(f.Pos)
		s := z * math.Min(1, float64(f.Age+1)/30)
		a := 1.0
		if f.Age > 38000 {
			a = math.Max(0, float64(40000-f.Age)/2000)
		}
		c := foodColors[f.Kind%4]
		r.drawFoodItem(x, y, s, f.Rot, f.Kind, a)
		if f.Kind == 3 {
			r.fxB.Glow(x, y, 5*s, float32(c[0]), float32(c[1]), float32(c[2]), float32(0.2+0.4*r.night))
		}
	}
}

func (r *Renderer) drawFoodItem(x, y, s, rot float64, kind uint8, a float64) {
	c := foodColors[kind%4]
	r.shadowB.Circle(x+s, y+s, 2.6*s, c4(0, 0, 0, 0.25*a))
	switch kind {
	case 0:
		r.sceneB.Ellipse(x, y, 2.4*s, 1.5*s, rot, c4(c[0], c[1], c[2], a))
		r.sceneB.Ellipse(x-0.4*s, y-0.4*s, 1.2*s, 0.6*s, rot, c4(1, 0.95, 0.75, 0.6*a))
	case 2:
		r.sceneB.Ellipse(x, y, 2.6*s, 2*s, rot, c4(c[0], c[1], c[2], a))
		r.sceneB.Circle(x+0.5*s, y-0.3*s, 1*s, c4(0.8, 0.5, 0.55, 0.6*a))
	default:
		r.sceneB.Circle(x, y, 2*s, c4(c[0], c[1], c[2], a))
		r.sceneB.Circle(x-0.6*s, y-0.6*s, 0.7*s, c4(1, 1, 1, 0.7*a))
	}
}

func (r *Renderer) drawAnt(w *sim.World, a *sim.Ant, shx, shy float64) {
	if !r.visible(a.Pos, 20) {
		return
	}
	z := r.cam.Z
	x, y := r.ws(a.Pos)
	s := a.Size * z * 1.6
	h := a.Heading
	fx, fy := math.Cos(h), math.Sin(h)
	px, py := -fy, fx
	col := w.Colonies[a.Colony].Color
	base := fxTint(sim.Mixc([3]float64{0.1, 0.07, 0.06}, col, 0.3), &a.Fx)
	if a.Flash > 0.05 {
		base = sim.Mixc(base, [3]float64{1, 1, 1}, a.Flash*0.8)
	}
	bc := c4(base[0], base[1], base[2], 1)
	hl := c4(math.Min(1, base[0]*2+0.15), math.Min(1, base[1]*2+0.12), math.Min(1, base[2]*2+0.1), 0.45)
	at := func(f, p float64) (float64, float64) { return x + fx*f*s + px*p*s, y + fy*f*s + py*p*s }

	r.shadowB.Quad(SprGlow, x+shx*s*0.8, y+shy*s*0.8, 6*s, 3.4*s, h, 0, 0, 0, 0.35)
	ec := sim.ElemColors[a.Element]
	glowA := float32((0.1 + 0.22*r.night) * (0.3 + 0.7*a.Mana))

	if s < 1.0 {
		r.sceneB.Circle(x, y, math.Max(1.3, 2.2*s), bc)
		ax, ay := at(-2.6, 0)
		r.fxB.Glow(ax, ay, 3.5*s+1.5, float32(ec[0]), float32(ec[1]), float32(ec[2]), glowA)
		if a.Carrying {
			cx, cy := at(2.5, 0)
			fc := foodColors[a.CarryK%4]
			r.sceneB.Circle(cx, cy, math.Max(1, 1.2*s), c4(fc[0], fc[1], fc[2], 1))
		}
		return
	}

	// Legs: alternating tripod gait
	if s >= 1.3 {
		lw := math.Max(0.7, 0.42*s)
		lc := c4(base[0]*0.8, base[1]*0.8, base[2]*0.8, 1)
		for i := -1; i <= 1; i++ {
			for _, side := range []float64{-1, 1} {
				group := 0.0
				if (i+2+int(side+1)/2)%2 == 0 {
					group = math.Pi
				}
				sw := math.Sin(a.Gait*1.15+group) * 1.3
				ax, ay := at(float64(i)*0.8, 0)
				kx, ky := at(float64(i)*1.4+sw*0.5, side*2.3)
				tx, ty := at(float64(i)*2.5+sw, side*3.9)
				r.sceneB.Line(SprDot, ax, ay, kx, ky, lw, lc)
				r.sceneB.Line(SprDot, kx, ky, tx, ty, lw*0.8, lc)
			}
		}
	}
	// Body
	ax, ay := at(-2.7, 0)
	r.sceneB.Ellipse(ax, ay, 2.4*s, 1.8*s, h, bc)
	hx, hy := at(-3.0, -0.6)
	r.sceneB.Ellipse(hx, hy, 1.1*s, 0.6*s, h, hl)
	r.fxB.Glow(ax, ay, 2.6*s, float32(ec[0]), float32(ec[1]), float32(ec[2]), glowA)
	pcx, pcy := at(-1.1, 0)
	r.sceneB.Circle(pcx, pcy, 0.6*s, bc)
	tx, ty := at(0.2, 0)
	r.sceneB.Ellipse(tx, ty, 1.35*s, 0.9*s, h, bc)
	headR := 1.1 * (0.85 + 0.35*(a.Size-0.7))
	hdx, hdy := at(1.9, 0)
	r.sceneB.Circle(hdx, hdy, headR*s, bc)
	r.sceneB.Circle(hdx-fx*0.2*s-px*0.4*s, hdy-fy*0.2*s-py*0.4*s, headR*0.4*s, hl)
	// Antennae
	wob := math.Sin(float64(a.Age)*0.21+float64(a.ID)) * 0.4
	for _, side := range []float64{-1, 1} {
		ex, ey := at(2.9, side*0.4)
		mx, my := at(3.9+wob*side*0.3, side*1.3)
		qx, qy := at(5.0, side*(1.0+wob))
		lw := math.Max(0.6, 0.3*s)
		r.sceneB.Line(SprDot, ex, ey, mx, my, lw, bc)
		r.sceneB.Line(SprDot, mx, my, qx, qy, lw, bc)
	}
	// Mandibles for big soldiers
	if a.Size > 1.25 {
		for _, side := range []float64{-1, 1} {
			ex, ey := at(2.6, side*0.5)
			qx, qy := at(3.6, side*0.1)
			r.sceneB.Line(SprDot, ex, ey, qx, qy, math.Max(0.7, 0.5*s), bc)
		}
	}
	if a.Carrying {
		cx, cy := at(3.3, 0)
		r.drawFoodItem(cx, cy, s*0.6, h, a.CarryK, 1)
	}
	// Shopping errand and insecticide can, carried on the back
	switch {
	case a.Money:
		cx, cy := at(-1.2, 0)
		r.sceneB.Circle(cx, cy, 1.3*s, c4(0.95, 0.75, 0.2, 1))
		r.fxB.Quad(SprStar, cx-0.4*s, cy-0.4*s, 1.2*s, 1.2*s, r.time*2, 1, 0.9, 0.5, 0.6)
	case a.Pack:
		cx, cy := at(-1.2, 0)
		r.sceneB.Quad(SprSolid, cx, cy, 1.9*s, 1.5*s, h, 0.95, 0.95, 0.92, 1)
		r.sceneB.Quad(SprSolid, cx, cy, 1.9*s, 0.4*s, h, 0.1, 0.33, 0.72, 1)
	}
	if a.Can > 0 {
		cx, cy := at(-1.4, 0)
		r.sceneB.Quad(SprSolid, cx, cy, 0.9*s, 1.5*s, h, 0.85, 0.15, 0.12, 1)
		r.sceneB.Quad(SprSolid, cx, cy-0.9*s, 0.5*s, 0.3*s, h, 0.9, 0.9, 0.9, 1)
	}
	r.drawAuras(&a.Fx, x, y, 4.4*s, h, a.ID)
	// Earth shield: orbiting pebbles on top of the bubble
	if a.Fx.Has(sim.FxShield) {
		fa := float32(math.Min(1, float64(a.Fx[sim.FxShield])/40))
		for k := range 3 {
			ang := r.time*3 + float64(k)*2.09 + float64(a.ID)
			ox, oy := x+math.Cos(ang)*7*s, y+math.Sin(ang)*5*s
			r.topB.Circle(ox, oy, 1.1*s, c4(0.5, 0.42, 0.32, float64(fa)))
		}
	}
	// Casting flare
	if since := w.Tick - a.LastCast; a.LastCast > 0 && since < 20 {
		f := float32(1 - float64(since)/20)
		r.fxB.Glow(x, y, 12*s, float32(ec[0]), float32(ec[1]), float32(ec[2]), 0.8*f)
		r.lightB.Glow(x, y, 60*s, float32(ec[0]), float32(ec[1]), float32(ec[2]), 0.8*f)
	}
	if r.night > 0.2 {
		r.lightB.Glow(x, y, 14*s, float32(ec[0]), float32(ec[1]), float32(ec[2]), glowA*0.25)
	}
}

func (r *Renderer) drawMonster(w *sim.World, m *sim.Monster, shx, shy float64) {
	if !r.visible(m.Pos, 60+float64(len(m.Segments))*11) {
		return
	}
	z := r.cam.Z
	x, y := r.ws(m.Pos)
	sp := 0.4 + 0.6*m.Spawn
	s := z * sp
	h := m.Heading
	fx, fy := math.Cos(h), math.Sin(h)
	px, py := -fy, fx
	at := func(f, p float64) (float64, float64) { return x + fx*f*s + px*p*s, y + fy*f*s + py*p*s }
	tint := func(c [3]float64) col4 {
		c = fxTint(c, &m.Fx)
		if m.Flash > 0.05 {
			c = sim.Mixc(c, [3]float64{1, 1, 1}, m.Flash*0.85)
		}
		return c4(c[0], c[1], c[2], sp)
	}
	lunge := m.Lunge * 3
	x += fx * lunge * s
	y += fy * lunge * s

	switch m.Kind {
	case sim.MonSpider:
		body := [3]float64{0.13, 0.08, 0.12}
		r.shadowB.Quad(SprGlow, x+shx*3*s, y+shy*3*s, 18*s, 14*s, h, 0, 0, 0, 0.4)
		lc := tint([3]float64{0.1, 0.06, 0.09})
		for i := range 4 {
			for _, side := range []float64{-1, 1} {
				ph := math.Sin(m.Gait*0.9 + float64(i)*1.6 + side*1.2)
				base := float64(i)*1.3 - 1.5
				ax, ay := at(base*0.6, side*2)
				kx, ky := at(base*2.2+ph*1.5, side*9)
				tx, ty := at(base*4.2+ph*3, side*14)
				r.sceneB.Line(SprDot, ax, ay, kx, ky, 1.6*s, lc)
				r.sceneB.Line(SprDot, kx, ky, tx, ty, 1.2*s, lc)
			}
		}
		ax, ay := at(-6, 0)
		r.sceneB.Ellipse(ax, ay, 7.5*s, 6*s, h, tint(body))
		mx, my := at(-6.5, 0)
		r.sceneB.Ellipse(mx, my, 3*s, 1.2*s, h, tint([3]float64{0.6, 0.15, 0.2}))
		cx, cy := at(1.5, 0)
		r.sceneB.Ellipse(cx, cy, 4.5*s, 4*s, h, tint(body))
		for _, side := range []float64{-1, 1} {
			ex, ey := at(4.5, side*1.4)
			r.fxB.Glow(ex, ey, 2.2*s, 1, 0.1, 0.1, 0.9)
		}
	case sim.MonBeetle:
		body := [3]float64{0.08, 0.18, 0.16}
		r.shadowB.Quad(SprGlow, x+shx*4*s, y+shy*4*s, 20*s, 15*s, h, 0, 0, 0, 0.45)
		lc := tint([3]float64{0.05, 0.08, 0.08})
		for i := -1; i <= 1; i++ {
			for _, side := range []float64{-1, 1} {
				ph := math.Sin(m.Gait*1.1+float64(i)*math.Pi+side*math.Pi/2) * 2
				ax, ay := at(float64(i)*3, side*5)
				kx, ky := at(float64(i)*5+ph, side*11)
				tx, ty := at(float64(i)*7+ph*1.5, side*15)
				r.sceneB.Line(SprDot, ax, ay, kx, ky, 2*s, lc)
				r.sceneB.Line(SprDot, kx, ky, tx, ty, 1.5*s, lc)
			}
		}
		ex, ey := at(-2, 0)
		r.sceneB.Ellipse(ex, ey, 12*s, 9*s, h, tint(body))
		// Iridescent elytra sheen
		shine := 0.5 + 0.5*math.Sin(r.time*1.5+m.Pos.X*0.02)
		hx, hy := at(-3, -3)
		r.sceneB.Ellipse(hx, hy, 6*s, 2.5*s, h, c4(0.3+0.3*shine, 0.6, 0.55-0.2*shine, 0.35*sp))
		sx0, sy0 := at(-12, 0)
		sx1, sy1 := at(6, 0)
		r.sceneB.Line(SprDot, sx0, sy0, sx1, sy1, 0.9*s, c4(0.02, 0.05, 0.05, 0.8))
		px2, py2 := at(8, 0)
		r.sceneB.Ellipse(px2, py2, 5*s, 6*s, h, tint([3]float64{0.06, 0.14, 0.12}))
		hx2, hy2 := at(13, 0)
		r.sceneB.Circle(hx2, hy2, 3.2*s, tint([3]float64{0.05, 0.1, 0.09}))
		tx, ty := at(18, 0)
		r.sceneB.Line(SprDot, hx2, hy2, tx, ty, 2*s, tint([3]float64{0.1, 0.2, 0.18}))
	case sim.MonWasp:
		bob := math.Sin(r.time*6+float64(m.ID)) * 2
		r.shadowB.Quad(SprGlow, x+shx*10*s+14*s, y+shy*10*s+18*s, 10*s, 6*s, h, 0, 0, 0, 0.3)
		y -= (6 + bob) * s
		for i := range 4 {
			f := -4 - float64(i)*2.4
			ax, ay := at(f, 0)
			c := [3]float64{0.95, 0.75, 0.1}
			if i%2 == 1 {
				c = [3]float64{0.1, 0.08, 0.05}
			}
			rr := 3.8 - float64(i)*0.55
			r.sceneB.Ellipse(ax, ay, rr*s*1.1, rr*s, h, tint(c))
		}
		sx, sy := at(-14.5, 0)
		r.sceneB.Line(SprDot, sx, sy, sx-fx*2*s, sy-fy*2*s, 0.9*s, tint([3]float64{0.1, 0.08, 0.05}))
		tx, ty := at(0, 0)
		r.sceneB.Ellipse(tx, ty, 3.2*s, 2.8*s, h, tint([3]float64{0.25, 0.18, 0.08}))
		hx, hy := at(3.8, 0)
		r.sceneB.Circle(hx, hy, 2.4*s, tint([3]float64{0.9, 0.7, 0.1}))
		flap := math.Sin(r.time*55+float64(m.ID)) * 0.5
		for _, side := range []float64{-1, 1} {
			wx, wy := at(-1, side*5)
			r.topB.Quad(SprDot, wx, wy, 6*s, 2.6*s, h+side*(0.9+flap), 0.9, 0.95, 1, 0.3)
		}
	case sim.MonCentipede:
		n := len(m.Segments)
		for i := n - 1; i >= 0; i-- {
			seg := m.Segments[i]
			sx, sy := r.ws(seg)
			prev := m.Pos
			if i > 0 {
				prev = m.Segments[i-1]
			}
			d := prev.Sub(seg)
			ang := math.Atan2(d.Y, d.X)
			rr := (11 - 5*float64(i)/float64(n)) * s
			ph := math.Sin(m.Gait*1.4 - float64(i)*0.7)
			for _, side := range []float64{-1, 1} {
				la := ang + side*(1.5+ph*0.35)
				kx, ky := sx+math.Cos(la)*rr*1.6, sy+math.Sin(la)*rr*1.6
				tx, ty := sx+math.Cos(la-side*0.5)*rr*2.6, sy+math.Sin(la-side*0.5)*rr*2.6
				lc := tint([3]float64{0.75, 0.45, 0.12})
				r.sceneB.Line(SprDot, sx, sy, kx, ky, 1.8*s, lc)
				r.sceneB.Line(SprDot, kx, ky, tx, ty, 1.3*s, lc)
			}
			r.shadowB.Circle(sx+shx*4*s, sy+shy*4*s, rr*1.4, c4(0, 0, 0, 0.3))
			c := [3]float64{0.5, 0.14, 0.05}
			if i%2 == 1 {
				c = [3]float64{0.28, 0.07, 0.04}
			}
			r.sceneB.Ellipse(sx, sy, rr*1.1, rr, ang, tint(c))
			r.sceneB.Ellipse(sx-math.Sin(ang)*rr*0.35, sy+math.Cos(ang)*rr*-0.35, rr*0.55, rr*0.28, ang, c4(1, 0.6, 0.35, 0.35*sp))
			pulse := float32(0.5 + 0.5*math.Sin(r.time*4-float64(i)*0.5))
			r.fxB.Glow(sx, sy, rr*0.55, 1, 0.75, 0.2, 0.5*pulse)
		}
		r.shadowB.Circle(x+shx*4*s, y+shy*4*s, 20*s, c4(0, 0, 0, 0.35))
		r.sceneB.Ellipse(x, y, 14*s, 12*s, h, tint([3]float64{0.42, 0.1, 0.04}))
		hx, hy := at(-2, -4)
		r.sceneB.Ellipse(hx, hy, 6*s, 3*s, h, c4(1, 0.6, 0.35, 0.3*sp))
		for _, side := range []float64{-1, 1} {
			bx, by := at(10, side*5)
			tx, ty := at(20, side*(2+math.Sin(r.time*4)*2.5))
			r.sceneB.Line(SprDot, bx, by, tx, ty, 3*s, tint([3]float64{0.2, 0.05, 0.02}))
			ax0, ay0 := at(9, side*3)
			ax1, ay1 := at(30, side*(14+math.Sin(r.time*2+side)*4))
			r.sceneB.Line(SprDot, ax0, ay0, ax1, ay1, 1.4*s, tint([3]float64{0.6, 0.3, 0.1}))
			ex, ey := at(6, side*5)
			r.fxB.Glow(ex, ey, 4.5*s, 1, 0.85, 0.15, 1)
			r.lightB.Glow(ex, ey, 60*s, 1, 0.5, 0.1, 0.6)
		}
	case sim.MonGolem:
		r.shadowB.Circle(x+shx*6*s, y+shy*6*s, 32*s, c4(0, 0, 0, 0.45))
		stomp := math.Sin(m.Gait * 0.8)
		for _, side := range []float64{-1, 1} {
			ax, ay := at(2+stomp*side*4, side*22)
			r.sceneB.Circle(ax, ay, 9*s, tint([3]float64{0.25, 0.22, 0.2}))
			r.fxB.Glow(ax, ay, 6*s, 1, 0.4, 0.05, 0.5)
		}
		r.sceneB.Circle(x, y, 22*s, tint([3]float64{0.3, 0.26, 0.24}))
		rng := rand.New(rand.NewSource(int64(m.ID)))
		for i := range 7 {
			a := rng.Float64() * 6.28
			d := rng.Float64() * 13
			r.sceneB.Circle(x+math.Cos(a)*d*s, y+math.Sin(a)*d*s, (5+rng.Float64()*5)*s, tint([3]float64{0.36, 0.31, 0.28}))
			_ = i
		}
		pulse := float32(0.7 + 0.3*math.Sin(r.time*3))
		r.fxB.Glow(x, y, 16*s, 1, 0.45, 0.08, 0.9*pulse)
		r.fxB.Quad(SprCrack, x, y, 22*s, 22*s, float64(m.ID), 1, 0.5, 0.1, 0.9*pulse)
		ex, ey := at(12, 0)
		r.fxB.Glow(ex, ey, 5*s, 1, 0.9, 0.3, 1)
		r.lightB.Glow(x, y, 170*s, 1, 0.4, 0.1, 0.9*pulse)
		if rand.Float64() < 0.4 {
			r.parts.Add(Particle{X: m.Pos.X + rnd(-15, 15), Y: m.Pos.Y + rnd(-15, 15), VZ: rnd(0.4, 1), Max: 50, Size: rnd(1.5, 3), R: 1, G: 0.6, B: 0.2, R2: 1, G2: 0.1, B2: 0, A: 1, Spr: SprGlow, Add: true})
		}
	}
	if m.Fx.Has(sim.FxBurn) {
		r.lightB.Glow(x, y, 60*s, 1, 0.5, 0.1, 0.6)
	}
	r.drawAuras(&m.Fx, x, y, m.Radius*s, h, m.ID)
}

func (r *Renderer) drawFireballs(w *sim.World) {
	z := r.cam.Z
	for _, b := range w.Balls {
		if !r.visible(b.Pos, 60) {
			continue
		}
		for i, tp := range b.Trail {
			tx, ty := r.ws(tp)
			f := 1 - float64(i)/float64(len(b.Trail))
			r.fxB.Glow(tx, ty, (4+6*f)*z, 1, float32(0.3+0.4*f), 0.05, float32(0.5*f))
		}
		x, y := r.ws(b.Pos)
		fl := float32(0.85 + 0.15*rand.Float64())
		r.fxB.Glow(x, y, 16*z, 1, 0.5, 0.1, 0.8*fl)
		r.fxB.Glow(x, y, 6*z, 1, 0.95, 0.8, 1)
		r.lightB.Glow(x, y, 150*z, 1, 0.55, 0.15, 1.1*fl)
	}
}

func (r *Renderer) drawBolts() {
	z := r.cam.Z
	for i := range r.bolts {
		b := &r.bolts[i]
		if b.Jag == nil || r.frame%2 == 0 {
			b.Jag = b.Jag[:0]
			for k := 0; k+1 < len(b.Pts); k++ {
				a, c := b.Pts[k], b.Pts[k+1]
				d := c.Sub(a)
				l := d.Len()
				nx, ny := -d.Y/math.Max(l, 1), d.X/math.Max(l, 1)
				seg := 8
				for j := range seg {
					f := float64(j) / float64(seg)
					off := 0.0
					if j > 0 {
						off = rnd(-1, 1) * l * 0.09
					}
					b.Jag = append(b.Jag, a.Add(d.Scale(f)).Add(sim.V(nx*off, ny*off)))
				}
			}
			b.Jag = append(b.Jag, b.Pts[len(b.Pts)-1])
		}
		f := b.Life / b.Max
		fl := float32(f) * float32(0.6+0.4*rand.Float64())
		for k := 0; k+1 < len(b.Jag); k++ {
			x0, y0 := r.ws(b.Jag[k])
			x1, y1 := r.ws(b.Jag[k+1])
			wd := b.Width * z
			r.fxB.Line(SprGlow, x0, y0, x1, y1, 16*wd, col4{b.Col[0], b.Col[1], b.Col[2], 0.35 * fl})
			r.fxB.Line(SprGlow, x0, y0, x1, y1, 5*wd, col4{b.Col[0], b.Col[1], b.Col[2], 0.8 * fl})
			r.fxB.Line(SprDot, x0, y0, x1, y1, 1.6*wd, col4{1, 1, 1, fl})
			if k%3 == 0 {
				r.lightB.Glow(x0, y0, 90*z*b.Width, b.Col[0], b.Col[1], b.Col[2], 0.7*fl)
			}
			// Occasional branch
			if k%4 == 1 && rand.Float64() < 0.4 {
				ang := rnd(0, 6.28)
				l := rnd(8, 22) * z * b.Width
				r.fxB.Line(SprGlow, x0, y0, x0+math.Cos(ang)*l, y0+math.Sin(ang)*l, 3*wd, col4{b.Col[0], b.Col[1], b.Col[2], 0.6 * fl})
			}
		}
	}
}

func (r *Renderer) drawParticles() {
	z := r.cam.Z
	for i := range r.parts.list {
		p := &r.parts.list[i]
		x, y := r.ws(sim.V(p.X, p.Y-p.Z))
		if x < -80 || y < -80 || x > SW+80 || y > SH+80 {
			continue
		}
		t := 1 - p.Life/p.Max
		fade := sim.ClampF(p.Life/(p.Max*0.35), 0, 1)
		a := p.A * float32(fade)
		if p.Flicker {
			a *= float32(0.35 + 0.65*math.Max(0, math.Sin(p.Life*0.12+p.X)))
		}
		cr, cg, cb := lerp32(p.R, p.R2, t), lerp32(p.G, p.G2, t), lerp32(p.B, p.B2, t)
		sz := p.Size * z
		hw, hh, rot := sz, sz, p.Rot
		if p.Stretch > 0 {
			v := math.Hypot(p.VX, p.VY-p.VZ)
			hw = sz * (1 + v*p.Stretch)
			rot = math.Atan2(p.VY-p.VZ, p.VX)
		}
		if p.Spr == SprShard || p.Spr == SprLeaf {
			hh = sz * 0.45
		}
		if p.Add {
			r.fxB.Quad(int(p.Spr), x, y, hw, hh, rot, cr, cg, cb, a)
		} else {
			if p.Z > 1 && p.Grav > 0 {
				sx, sy := r.ws(sim.V(p.X, p.Y))
				r.shadowB.Circle(sx+1, sy+1, sz, c4(0, 0, 0, 0.25*float64(fade)))
			}
			r.topB.Quad(int(p.Spr), x, y, hw, hh, rot, cr, cg, cb, a)
		}
		if p.Light > 0 {
			r.lightB.Glow(x, y, sz*6+10*z, cr, cg, cb, p.Light*float32(fade))
		}
	}
}

func (r *Renderer) drawRain() {
	if r.wet < 0.03 {
		return
	}
	n := int(r.wet * 500)
	ang := 1.75
	dx, dy := math.Cos(ang), math.Sin(ang)
	for range n {
		x, y := rnd(-50, SW+50), rnd(-50, SH+50)
		l := rnd(14, 30)
		r.topB.Line(SprGlow, x, y, x+dx*l, y+dy*l, 1.4, c4(0.75, 0.82, 0.95, 0.22*r.wet))
	}
}
