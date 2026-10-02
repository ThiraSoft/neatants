package render

import (
	"github.com/ThiraSoft/neatants/internal/sim"
	"math"

	"github.com/hajimehoshi/ebiten/v2"
)

// Simplified view: the minimap look, full screen, following the camera.

const simpleDiv = 4 // world units per background pixel

func (r *Renderer) DrawSimple(w *sim.World) *ebiten.Image {
	if r.simpleBG == nil || r.simpleAt != w.Cave {
		pw, ph := sim.WorldW/simpleDiv, sim.WorldH/simpleDiv
		if r.simpleBG == nil {
			r.simpleBG = ebiten.NewImage(pw, ph)
		}
		r.simpleBG.WritePixels(mapPix(w, pw, ph))
		r.simpleAt = w.Cave
	}
	z := r.cam.Z
	dst := r.final
	dst.Fill(colorRGB(0.08, 0.08, 0.07))
	op := &ebiten.DrawImageOptions{Filter: ebiten.FilterLinear}
	op.GeoM.Scale(simpleDiv*z, simpleDiv*z)
	x0, y0 := r.ws(sim.Vec2{})
	op.GeoM.Translate(x0, y0)
	op.ColorScale.Scale(0.85, 0.85, 0.85, 1)
	dst.DrawImage(r.simpleBG, op)

	b := &r.topB
	px := func(world, min float64) float64 { return math.Max(world*z, min) }
	dot := func(p sim.Vec2, rad float64, c [4]float32) {
		if r.visible(p, 40) {
			x, y := r.ws(p)
			b.Circle(x, y, rad, c)
		}
	}

	for i := range w.Foods {
		f := &w.Foods[i]
		if !f.Taken {
			c := foodColors[f.Kind%4]
			dot(f.Pos, px(3, 1), c4(c[0], c[1], c[2], 0.8))
		}
	}
	for _, c := range w.Colonies {
		if !c.Alive {
			continue
		}
		x, y := r.ws(c.Pos)
		b.Glow(x, y, px(140, 14), float32(c.Color[0]), float32(c.Color[1]), float32(c.Color[2]), 0.5)
		b.Circle(x, y, px(40, 4), c4(c.Color[0], c.Color[1], c.Color[2], 1))
	}
	for _, c := range w.Colonies {
		if s := w.SpouseOf(c.ID); s >= 0 && c.ID < s {
			o := w.Colonies[s]
			ax, ay := r.ws(c.Pos)
			bx, by := r.ws(o.Pos)
			b.Line(SprGlow, ax, ay, bx, by, px(30, 3), c4(1, 0.45, 0.65, 0.7))
			hs := px(50, 5)
			b.Quad(SprHeart, (ax+bx)/2, (ay+by)/2, hs, hs, 0, 1, 0.45, 0.65, 1)
		}
	}
	for _, a := range w.Ants {
		if a.Alive {
			c := w.Colonies[a.Colony].Color
			dot(a.Pos, px(6, 1), c4(c[0], c[1], c[2], 0.9))
		}
	}
	for _, m := range w.Monsters {
		if m.Alive {
			rad := 14.0
			if m.Kind >= sim.MonCentipede {
				rad = 32
			}
			dot(m.Pos, px(rad, 1.8), c4(1, 0.2, 0.15, 1))
		}
	}
	if sim.Cfg.ShopEnabled {
		x, y := r.ws(w.Shop)
		b.Quad(SprSolid, x, y, px(40, 4), px(30, 3), 0, 0.3, 0.55, 1, 1)
		if k := w.Kiosk; k != nil && k.Alive {
			c := w.Colonies[k.Owner].Color
			x, y := r.ws(k.Pos)
			s := px(25, 2.5)
			b.Quad(SprSolid, x, y, s, s, 0.785, float32(c[0]), float32(c[1]), float32(c[2]), 1)
		}
	}
	for _, f := range w.Frogs {
		dot(f.Pos, px(10, 1.2), c4(0.45, 0.95, 0.35, 0.9))
	}
	for _, k := range w.Couriers {
		dot(k.Pos, px(12, 1.6), c4(1, 1, 1, 1))
	}
	cx, cy := r.ws(w.Cave)
	b.Glow(cx, cy, px(170, 18), 1, 0.2, 0.1, float32(0.4+0.2*math.Sin(r.time*3)))
	b.Flush(dst, ebiten.BlendSourceOver)
	return dst
}
