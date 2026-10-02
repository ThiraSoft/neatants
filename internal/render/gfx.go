package render

import (
	"github.com/ThiraSoft/neatants/internal/sim"
	"image"
	"image/color"
	"math"
	"math/rand"

	"github.com/hajimehoshi/ebiten/v2"
)

// Sprite regions in the procedural atlas.
const (
	SprDot = iota
	SprGlow
	SprRing
	SprSmoke
	SprShard
	SprLeaf
	SprStar
	SprBubble
	SprScorch
	SprFrost
	SprCrack
	SprSplat
	SprRune
	SprSoft
	SprSolid
	SprHeart
	NumSprites
)

const atlasCell = 64

type spriteRect struct{ x, y, w, h float32 }

var (
	atlas    *ebiten.Image
	sprRects [NumSprites]spriteRect
	whiteImg *ebiten.Image
	whiteSub *ebiten.Image
)

func makeUnmanaged(w, h int) *ebiten.Image {
	return ebiten.NewImageWithOptions(image.Rect(0, 0, w, h), &ebiten.NewImageOptions{Unmanaged: true})
}

// buildAtlas renders every sprite procedurally into one texture so all
// quads can be batched into a handful of draw calls.
func buildAtlas() {
	const aw, ah = 512, 512
	pix := make([]byte, aw*ah*4)
	put := func(ox, oy, x, y int, r, g, b, a float64) {
		a = sim.ClampF(a, 0, 1)
		i := ((oy+y)*aw + ox + x) * 4
		pix[i] = byte(sim.ClampF(r, 0, 1) * a * 255)
		pix[i+1] = byte(sim.ClampF(g, 0, 1) * a * 255)
		pix[i+2] = byte(sim.ClampF(b, 0, 1) * a * 255)
		pix[i+3] = byte(a * 255)
	}
	cellFn := func(idx, size int, f func(u, v float64) (float64, float64, float64, float64)) {
		var ox, oy int
		if size == atlasCell {
			ox, oy = (idx%8)*atlasCell, (idx/8)*atlasCell
		} else {
			ox, oy = 0, 256
		}
		for y := range size {
			for x := range size {
				u := (float64(x)+0.5)/float64(size)*2 - 1
				v := (float64(y)+0.5)/float64(size)*2 - 1
				r, g, b, a := f(u, v)
				put(ox, oy, x, y, r, g, b, a)
			}
		}
		sprRects[idx] = spriteRect{float32(ox) + 0.5, float32(oy) + 0.5, float32(size) - 1, float32(size) - 1}
	}
	vn := func(x, y float64) float64 { return valueNoise(x, y) }

	cellFn(SprDot, 64, func(u, v float64) (float64, float64, float64, float64) {
		d := math.Hypot(u, v)
		return 1, 1, 1, sim.ClampF((1-d)*20, 0, 1)
	})
	cellFn(SprGlow, 64, func(u, v float64) (float64, float64, float64, float64) {
		d := math.Hypot(u, v)
		a := math.Exp(-d*d*4.5) * sim.ClampF((1-d)*6, 0, 1)
		return 1, 1, 1, a
	})
	cellFn(SprRing, 64, func(u, v float64) (float64, float64, float64, float64) {
		d := math.Hypot(u, v)
		x := (d - 0.85) / 0.08
		return 1, 1, 1, math.Exp(-x * x)
	})
	cellFn(SprSmoke, 64, func(u, v float64) (float64, float64, float64, float64) {
		d := math.Hypot(u, v)
		n := vn(u*3+5, v*3+2)*0.6 + vn(u*7, v*7)*0.4
		a := sim.ClampF((1-d)*1.6, 0, 1) * (0.4 + 0.6*n)
		return 1, 1, 1, a * a
	})
	cellFn(SprShard, 64, func(u, v float64) (float64, float64, float64, float64) {
		d := math.Abs(u)*3.2 + math.Abs(v)
		a := sim.ClampF((1-d)*8, 0, 1)
		hl := 0.8 + 0.2*(1-math.Abs(u)*3)
		return hl, hl, 1, a
	})
	cellFn(SprLeaf, 64, func(u, v float64) (float64, float64, float64, float64) {
		w := (1 - v*v) * 0.5
		a := sim.ClampF((w-math.Abs(u))*30, 0, 1) * sim.ClampF((1-math.Abs(v))*20, 0, 1)
		vein := 1 - 0.25*math.Exp(-u*u*400)
		return vein, vein, vein, a
	})
	cellFn(SprStar, 64, func(u, v float64) (float64, float64, float64, float64) {
		d := math.Hypot(u, v)
		s := math.Exp(-math.Abs(u)*14)*math.Exp(-math.Abs(v)*2.2) + math.Exp(-math.Abs(v)*14)*math.Exp(-math.Abs(u)*2.2)
		a := s + math.Exp(-d*d*30)
		return 1, 1, 1, sim.ClampF(a, 0, 1) * sim.ClampF((1-d)*4, 0, 1)
	})
	cellFn(SprBubble, 64, func(u, v float64) (float64, float64, float64, float64) {
		d := math.Hypot(u, v)
		a := math.Pow(sim.ClampF(d, 0, 1), 4)*0.8 + math.Exp(-math.Pow((d-0.93)/0.05, 2))
		return 1, 1, 1, sim.ClampF(a, 0, 1) * sim.ClampF((1-d)*30, 0, 1)
	})
	cellFn(SprScorch, 64, func(u, v float64) (float64, float64, float64, float64) {
		d := math.Hypot(u, v)
		n := vn(u*4+1, v*4+8)
		a := sim.ClampF((1-d-n*0.4)*2.2, 0, 1)
		return 0.05, 0.035, 0.03, a * 0.9
	})
	cellFn(SprFrost, 64, func(u, v float64) (float64, float64, float64, float64) {
		d := math.Hypot(u, v)
		ang := math.Atan2(v, u)
		spikes := math.Pow(math.Abs(math.Cos(ang*3)), 18)*0.7 + math.Pow(math.Abs(math.Cos(ang*6+0.5)), 30)*0.4
		a := sim.ClampF((spikes+0.25-d)*3, 0, 1) * sim.ClampF((1-d)*5, 0, 1)
		return 0.85, 0.95, 1, a * (0.6 + 0.4*vn(u*9, v*9))
	})
	cellFn(SprCrack, 64, func(u, v float64) (float64, float64, float64, float64) {
		d := math.Hypot(u, v)
		ang := math.Atan2(v, u) + vn(u*3, v*3)*0.8
		lines := math.Pow(math.Abs(math.Cos(ang*3.5)), 60)
		a := sim.ClampF(lines*1.5*(1-d)+math.Exp(-d*d*20)*0.8, 0, 1) * sim.ClampF((1-d)*6, 0, 1)
		return 0.08, 0.06, 0.05, a
	})
	cellFn(SprSplat, 64, func(u, v float64) (float64, float64, float64, float64) {
		d := math.Hypot(u, v)
		ang := math.Atan2(v, u)
		r := 0.45 + 0.25*vn(math.Cos(ang)*2+3, math.Sin(ang)*2+3) + 0.2*math.Pow(math.Abs(math.Sin(ang*5)), 8)
		a := sim.ClampF((r-d)*12, 0, 1)
		return 1, 1, 1, a
	})
	cellFn(SprHeart, 64, func(u, v float64) (float64, float64, float64, float64) {
		x, y := u*1.3, -v*1.3+0.25
		f := math.Pow(x*x+y*y-1, 3) - x*x*y*y*y
		return 1, 1, 1, sim.ClampF(-f*25, 0, 1)
	})
	cellFn(SprSolid, 64, func(u, v float64) (float64, float64, float64, float64) {
		e := math.Max(math.Abs(u), math.Abs(v))
		return 1, 1, 1, sim.ClampF((1-e)*32, 0, 1)
	})
	cellFn(SprSoft, 64, func(u, v float64) (float64, float64, float64, float64) {
		d := math.Hypot(u, v)
		return 1, 1, 1, sim.ClampF(1-d, 0, 1)
	})
	cellFn(SprRune, 128, func(u, v float64) (float64, float64, float64, float64) {
		d := math.Hypot(u, v)
		ang := math.Atan2(v, u)
		ring := func(r, w float64) float64 { x := (d - r) / w; return math.Exp(-x * x) }
		a := ring(0.92, 0.018) + ring(0.8, 0.012)*0.8 + ring(0.5, 0.01)*0.5
		// glyph ticks between the two outer rings
		seg := math.Mod(ang/(2*math.Pi)*24+24, 1)
		if d > 0.82 && d < 0.9 && seg > 0.35 && seg < 0.65 {
			a += 0.9 * sim.ClampF((0.15-math.Abs(seg-0.5))*12, 0, 1)
		}
		// hexagram
		for k := range 6 {
			t := float64(k) * math.Pi / 3
			nx, ny := math.Cos(t), math.Sin(t)
			dist := math.Abs(u*nx + v*ny - 0.4)
			a += math.Exp(-dist*dist/0.0002) * 0.55 * sim.ClampF((0.8-d)*10, 0, 1)
		}
		return 1, 1, 1, sim.ClampF(a, 0, 1)
	})

	atlas = ebiten.NewImage(aw, ah)
	atlas.WritePixels(pix)
	whiteImg = ebiten.NewImage(3, 3)
	whiteImg.Fill(image.White)
	whiteSub = whiteImg.SubImage(image.Rect(1, 1, 2, 2)).(*ebiten.Image)
}

func valueNoise(x, y float64) float64 {
	h := func(ix, iy int) float64 {
		n := uint32(ix*374761393 + iy*668265263)
		n = (n ^ (n >> 13)) * 1274126177
		return float64(n&0xffff) / 65535
	}
	ix, iy := int(math.Floor(x)), int(math.Floor(y))
	fx, fy := x-math.Floor(x), y-math.Floor(y)
	ux, uy := fx*fx*(3-2*fx), fy*fy*(3-2*fy)
	a, b := h(ix, iy), h(ix+1, iy)
	c, d := h(ix, iy+1), h(ix+1, iy+1)
	return a + (b-a)*ux + (c-a)*uy + (a-b-c+d)*ux*uy
}

// --- Sprite batch ---

type Batch struct {
	verts []ebiten.Vertex
	idx   []uint16
}

func (b *Batch) Reset() { b.verts, b.idx = b.verts[:0], b.idx[:0] }

// Quad adds a sprite centred on (x,y) with half extents (hw,hh) and rotation.
func (b *Batch) Quad(spr int, x, y, hw, hh, rot float64, r, g, bl, a float32) {
	if a <= 0.002 {
		return
	}
	c, s := math.Cos(rot), math.Sin(rot)
	ax, ay := c*hw, s*hw
	bx, by := -s*hh, c*hh
	b.quad4(spr,
		x-ax-bx, y-ay-by, x+ax-bx, y+ay-by,
		x-ax+bx, y-ay+by, x+ax+bx, y+ay+by, r, g, bl, a)
}

func (b *Batch) quad4(spr int, x0, y0, x1, y1, x2, y2, x3, y3 float64, r, g, bl, a float32) {
	if len(b.verts) > 65000 {
		return
	}
	sr := sprRects[spr]
	n := uint16(len(b.verts))
	b.verts = append(b.verts,
		ebiten.Vertex{DstX: float32(x0), DstY: float32(y0), SrcX: sr.x, SrcY: sr.y, ColorR: r, ColorG: g, ColorB: bl, ColorA: a},
		ebiten.Vertex{DstX: float32(x1), DstY: float32(y1), SrcX: sr.x + sr.w, SrcY: sr.y, ColorR: r, ColorG: g, ColorB: bl, ColorA: a},
		ebiten.Vertex{DstX: float32(x2), DstY: float32(y2), SrcX: sr.x, SrcY: sr.y + sr.h, ColorR: r, ColorG: g, ColorB: bl, ColorA: a},
		ebiten.Vertex{DstX: float32(x3), DstY: float32(y3), SrcX: sr.x + sr.w, SrcY: sr.y + sr.h, ColorR: r, ColorG: g, ColorB: bl, ColorA: a},
	)
	b.idx = append(b.idx, n, n+1, n+2, n+1, n+3, n+2)
}

func (b *Batch) Circle(x, y, rad float64, c [4]float32) {
	b.Quad(SprDot, x, y, rad, rad, 0, c[0], c[1], c[2], c[3])
}

func (b *Batch) Ellipse(x, y, rx, ry, rot float64, c [4]float32) {
	b.Quad(SprDot, x, y, rx, ry, rot, c[0], c[1], c[2], c[3])
}

func (b *Batch) Glow(x, y, rad float64, r, g, bl, a float32) {
	b.Quad(SprGlow, x, y, rad, rad, 0, r, g, bl, a)
}

// Line draws an anti-aliased segment using the centre column of a sprite.
func (b *Batch) Line(spr int, x0, y0, x1, y1, width float64, c [4]float32) {
	dx, dy := x1-x0, y1-y0
	l := math.Hypot(dx, dy)
	if l < 0.01 {
		return
	}
	nx, ny := -dy/l*width*0.5, dx/l*width*0.5
	if len(b.verts) > 65000 {
		return
	}
	sr := sprRects[spr]
	cx := sr.x + sr.w*0.5
	n := uint16(len(b.verts))
	mk := func(x, y float64, sy float32) ebiten.Vertex {
		return ebiten.Vertex{DstX: float32(x), DstY: float32(y), SrcX: cx, SrcY: sy, ColorR: c[0], ColorG: c[1], ColorB: c[2], ColorA: c[3]}
	}
	b.verts = append(b.verts,
		mk(x0+nx, y0+ny, sr.y), mk(x1+nx, y1+ny, sr.y),
		mk(x0-nx, y0-ny, sr.y+sr.h), mk(x1-nx, y1-ny, sr.y+sr.h))
	b.idx = append(b.idx, n, n+1, n+2, n+1, n+3, n+2)
}

func (b *Batch) Flush(dst *ebiten.Image, blend ebiten.Blend) {
	if len(b.idx) == 0 {
		return
	}
	op := &ebiten.DrawTrianglesOptions{Blend: blend, Filter: ebiten.FilterLinear}
	// Chunk so no call exceeds the index limit.
	const maxQuads = 10000
	for s := 0; s < len(b.idx); s += maxQuads * 6 {
		e := min(s+maxQuads*6, len(b.idx))
		vs := int(b.idx[s])
		ve := int(b.idx[e-1]) + 2
		ve = min(ve, len(b.verts))
		idx := make([]uint16, e-s)
		for i := range idx {
			idx[i] = b.idx[s+i] - uint16(vs)
		}
		dst.DrawTriangles(b.verts[vs:ve], idx, atlas, op)
	}
	b.Reset()
}

// --- Particles ---

type Particle struct {
	X, Y, Z    float64
	VX, VY, VZ float64
	Life, Max  float64
	Size, Grow float64
	Rot, VRot  float64
	Drag, Grav float64
	R, G, B, A float32
	R2, G2, B2 float32 // colour at end of life
	Spr        uint8
	Add        bool    // emissive (fx buffer) vs lit (scene)
	Stretch    float64 // elongate along velocity
	Light      float32 // contributes to light map
	Flicker    bool
	Bounce     bool
	Decal      int8 // decal sprite left on landing (-1 none)
}

type Particles struct {
	list []Particle
}

func (ps *Particles) Add(p Particle) {
	if len(ps.list) >= 16000 {
		return
	}
	if p.Max == 0 {
		p.Max = 60
	}
	p.Life = p.Max
	if p.R2 == 0 && p.G2 == 0 && p.B2 == 0 {
		p.R2, p.G2, p.B2 = p.R, p.G, p.B
	}
	if p.Decal == 0 {
		p.Decal = -1
	}
	ps.list = append(ps.list, p)
}

func (ps *Particles) Step(onLand func(p *Particle)) {
	live := ps.list[:0]
	for i := range ps.list {
		p := &ps.list[i]
		p.Life--
		if p.Life <= 0 {
			continue
		}
		p.X += p.VX
		p.Y += p.VY
		p.Z += p.VZ
		d := 1 - p.Drag
		p.VX *= d
		p.VY *= d
		p.VZ -= p.Grav
		if p.Z < 0 && p.Grav > 0 {
			p.Z = 0
			if p.Bounce && p.VZ < -0.6 {
				p.VZ *= -0.35
				p.VX *= 0.6
				p.VY *= 0.6
			} else {
				if p.Decal >= 0 && onLand != nil {
					onLand(p)
					p.Decal = -1
				}
				p.VZ = 0
				p.VX *= 0.8
				p.VY *= 0.8
			}
		}
		p.Rot += p.VRot
		p.Size += p.Grow
		if p.Size <= 0 {
			continue
		}
		live = append(live, *p)
	}
	ps.list = live
}

func rnd(a, b float64) float64 { return a + rand.Float64()*(b-a) }

func lerp32(a, b float32, t float64) float32 { return a + (b-a)*float32(t) }

func colorRGB(r, g, b float64) color.RGBA {
	return color.RGBA{uint8(r * 255), uint8(g * 255), uint8(b * 255), 255}
}

func sinf(x float64) float64 { return math.Sin(x) }
