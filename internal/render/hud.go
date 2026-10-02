package render

import (
	"bytes"
	"fmt"
	"github.com/ThiraSoft/neatants/internal/sim"
	"math"
	"sort"

	"github.com/hajimehoshi/ebiten/v2"
	"github.com/hajimehoshi/ebiten/v2/text/v2"
	"github.com/hajimehoshi/ebiten/v2/vector"
	"golang.org/x/image/font/gofont/gobold"
	"golang.org/x/image/font/gofont/gomedium"
	"golang.org/x/image/font/gofont/goregular"
	"golang.org/x/image/font/gofont/gosmallcaps"
)

type Fonts struct {
	Title, Big, H, Body, Small, Tiny *text.GoTextFace
}

func loadFonts() Fonts {
	src := func(b []byte) *text.GoTextFaceSource {
		s, err := text.NewGoTextFaceSource(bytes.NewReader(b))
		if err != nil {
			panic(err)
		}
		return s
	}
	caps, bold, med, reg := src(gosmallcaps.TTF), src(gobold.TTF), src(gomedium.TTF), src(goregular.TTF)
	return Fonts{
		Title: &text.GoTextFace{Source: caps, Size: 24},
		Big:   &text.GoTextFace{Source: caps, Size: 64},
		H:     &text.GoTextFace{Source: bold, Size: 15},
		Body:  &text.GoTextFace{Source: med, Size: 13},
		Small: &text.GoTextFace{Source: reg, Size: 12},
		Tiny:  &text.GoTextFace{Source: reg, Size: 10},
	}
}

type UI struct {
	caveAt   sim.Vec2
	f        Fonts
	b        Batch
	minimap  *ebiten.Image
	history  [sim.MaxColonies][]int
	histFrom [sim.MaxColonies]int
	netCache *netLayout
}

var (
	colText  = [3]float64{0.93, 0.92, 0.88}
	colMuted = [3]float64{0.62, 0.64, 0.66}
)

const (
	mmW, mmH = 300, 184
	mmX      = SW - mmW - 18
	mmY      = SH - mmH - 18
)

func NewUI(w *sim.World) *UI {
	u := &UI{f: loadFonts()}
	u.buildMinimap(w)
	return u
}

func (u *UI) buildMinimap(w *sim.World) {
	u.caveAt = w.Cave
	if u.minimap == nil {
		u.minimap = ebiten.NewImage(mmW, mmH)
	}
	u.minimap.WritePixels(mapPix(w, mmW, mmH))
}

// mapPix renders the schematic terrain (minimap style) at the given resolution.
func mapPix(w *sim.World, pw, ph int) []byte {
	pix := make([]byte, pw*ph*4)
	for y := range ph {
		for x := range pw {
			wx := float64(x) / float64(pw) * sim.WorldW
			wy := float64(y) / float64(ph) * sim.WorldH
			n := valueNoise(wx*0.004, wy*0.004)*0.6 + valueNoise(wx*0.012, wy*0.012)*0.4
			c := sim.Mixc([3]float64{0.3, 0.26, 0.18}, [3]float64{0.2, 0.32, 0.14}, smooth(0.35, 0.65, n))
			p := sim.V(wx, wy)
			if w.InPond(p, 0) {
				c = [3]float64{0.12, 0.3, 0.38}
			}
			if d := p.Sub(w.Cave).Len(); d < 260 {
				c = sim.Mixc(c, [3]float64{0.18, 0.12, 0.13}, smooth(260, 150, d))
				if d < 90 {
					c = [3]float64{0.35, 0.06, 0.04}
				}
			}
			for _, b := range w.Bushes {
				if p.Sub(b.Pos).Len() < b.R+10 {
					c = [3]float64{0.16, 0.36, 0.12}
				}
			}
			i := (y*pw + x) * 4
			pix[i], pix[i+1], pix[i+2], pix[i+3] = byte(c[0]*255), byte(c[1]*255), byte(c[2]*255), 255
		}
	}
	return pix
}

// --- Primitive helpers ---

func (u *UI) text(dst *ebiten.Image, s string, face *text.GoTextFace, x, y float64, c [3]float64, a float64, align text.Align) {
	op := &text.DrawOptions{}
	op.PrimaryAlign = align
	op.GeoM.Translate(x+1, y+1)
	op.ColorScale.Scale(0, 0, 0, float32(a*0.55))
	text.Draw(dst, s, face, op)
	op = &text.DrawOptions{}
	op.PrimaryAlign = align
	op.GeoM.Translate(x, y)
	op.ColorScale.Scale(float32(c[0]*a), float32(c[1]*a), float32(c[2]*a), float32(a))
	text.Draw(dst, s, face, op)
}

func roundRect(x, y, w, h, r float32) *vector.Path {
	p := &vector.Path{}
	p.MoveTo(x+r, y)
	p.LineTo(x+w-r, y)
	p.ArcTo(x+w, y, x+w, y+r, r)
	p.LineTo(x+w, y+h-r)
	p.ArcTo(x+w, y+h, x+w-r, y+h, r)
	p.LineTo(x+r, y+h)
	p.ArcTo(x, y+h, x, y+h-r, r)
	p.LineTo(x, y+r)
	p.ArcTo(x, y, x+r, y, r)
	p.Close()
	return p
}

func fillPath(dst *ebiten.Image, p *vector.Path, c [4]float32) {
	vs, is := p.AppendVerticesAndIndicesForFilling(nil, nil)
	for i := range vs {
		vs[i].SrcX, vs[i].SrcY = 1.5, 1.5
		vs[i].ColorR, vs[i].ColorG, vs[i].ColorB, vs[i].ColorA = c[0], c[1], c[2], c[3]
	}
	dst.DrawTriangles(vs, is, whiteImg, &ebiten.DrawTrianglesOptions{AntiAlias: true, FillRule: ebiten.NonZero})
}

func strokePath(dst *ebiten.Image, p *vector.Path, width float32, c [4]float32) {
	vs, is := p.AppendVerticesAndIndicesForStroke(nil, nil, &vector.StrokeOptions{Width: width})
	for i := range vs {
		vs[i].SrcX, vs[i].SrcY = 1.5, 1.5
		vs[i].ColorR, vs[i].ColorG, vs[i].ColorB, vs[i].ColorA = c[0], c[1], c[2], c[3]
	}
	dst.DrawTriangles(vs, is, whiteImg, &ebiten.DrawTrianglesOptions{AntiAlias: true})
}

// panel draws a frosted-glass card: blurred backdrop, tint, border, sheen.
func (u *UI) panel(dst *ebiten.Image, glass *ebiten.Image, x, y, w, h float64, accent [3]float64) {
	shadow := roundRect(float32(x+2), float32(y+5), float32(w), float32(h), 14)
	fillPath(dst, shadow, [4]float32{0, 0, 0, 0.28})
	p := roundRect(float32(x), float32(y), float32(w), float32(h), 12)
	vs, is := p.AppendVerticesAndIndicesForFilling(nil, nil)
	for i := range vs {
		vs[i].SrcX, vs[i].SrcY = vs[i].DstX/4, vs[i].DstY/4
		vs[i].ColorR, vs[i].ColorG, vs[i].ColorB, vs[i].ColorA = 1, 1, 1, 1
	}
	dst.DrawTriangles(vs, is, glass, &ebiten.DrawTrianglesOptions{AntiAlias: true, FillRule: ebiten.NonZero, Filter: ebiten.FilterLinear})
	fillPath(dst, p, [4]float32{0.045, 0.05, 0.07, 0.62})
	top := roundRect(float32(x+1), float32(y+1), float32(w-2), float32(math.Min(26, h/2)), 11)
	fillPath(dst, top, [4]float32{1, 1, 1, 0.03})
	strokePath(dst, p, 1, [4]float32{float32(accent[0]) * 0.5, float32(accent[1]) * 0.5, float32(accent[2]) * 0.5, 0.35})
}

func (u *UI) bar(x, y, w, h, frac float64, c [3]float64, bg float64) {
	frac = sim.ClampF(frac, 0, 1)
	u.b.Line(SprDot, x+h/2, y+h/2, x+w-h/2, y+h/2, h, c4(1, 1, 1, bg))
	if frac > 0.01 {
		u.b.Line(SprDot, x+h/2, y+h/2, x+h/2+(w-h)*frac, y+h/2, h, c4(c[0], c[1], c[2], 0.95))
		u.b.Line(SprGlow, x+h/2, y+h/2, x+h/2+(w-h)*frac, y+h/2, h*2.5, c4(c[0], c[1], c[2], 0.18))
	}
}

// --- Main HUD ---

func (u *UI) Draw(dst *ebiten.Image, g *Game) {
	w, r := g.World, g.R
	// Frosted glass source
	r.glassA.Clear()
	op := &ebiten.DrawImageOptions{Filter: ebiten.FilterLinear}
	op.GeoM.Scale(0.25, 0.25)
	r.glassA.DrawImage(r.final, op)
	r.blur(r.glassA, r.glassB, 1.4)
	r.blur(r.glassA, r.glassB, 2.5)

	if w.Cave != u.caveAt {
		u.buildMinimap(w)
	}
	u.worldLabels(dst, g)
	u.b.Flush(dst, ebiten.BlendSourceOver)

	if g.HideHUD {
		u.drawBanner(dst, r)
		return
	}
	u.header(dst, g)
	u.colonies(dst, g)
	u.wave(dst, g)
	u.chronicle(dst, g)
	u.minimapDraw(dst, g)
	if g.Selected != nil {
		u.inspector(dst, g, g.Selected)
	}
	if g.ShowHelp {
		u.help(dst, g)
	}
	u.drawBanner(dst, r)
	u.b.Flush(dst, ebiten.BlendSourceOver)
	_ = w
}

func (u *UI) worldLabels(dst *ebiten.Image, g *Game) {
	w, r := g.World, g.R
	z := r.cam.Z
	for _, c := range w.Colonies {
		if !c.Alive || !r.visible(c.Pos, 100) {
			continue
		}
		x, y := r.ws(c.Pos)
		if z > 0.28 {
			u.text(dst, c.Name, u.f.H, x, y-sim.NestRadius*2.3*z-26, c.Color, 0.9, text.AlignCenter)
			if w.SpouseOf(c.ID) >= 0 {
				hx := x + text.Advance(c.Name, u.f.H)/2 + 10
				u.b.Quad(SprHeart, hx, y-sim.NestRadius*2.3*z-17, 6, 6, 0, 1, 0.45, 0.65, 1)
			}
		}
		if c.HP < sim.Cfg.NestMaxHP*0.98 {
			u.bar(x-40, y-sim.NestRadius*2.3*z-6, 80, 5, c.HP/sim.Cfg.NestMaxHP, [3]float64{1, 0.35, 0.3}, 0.15)
		}
	}
	u.drawWTFLabels(g, func(s string, face *text.GoTextFace, x, y float64, c [3]float64, a float64) {
		u.text(dst, s, face, x, y, c, a, text.AlignCenter)
	})
	if sim.Cfg.ShopEnabled && r.visible(w.Shop, sim.ShopW) && z > 0.22 {
		x, y := r.ws(w.Shop)
		u.text(dst, "WALLMART", u.f.H, x, y-sim.ShopH*0.46*z-22, [3]float64{0.35, 0.6, 1}, 0.95, text.AlignCenter)
	}
	for _, m := range w.Monsters {
		if !m.Alive || m.HP >= m.MaxHP || !r.visible(m.Pos, 30) || m.Kind >= sim.MonCentipede {
			continue
		}
		x, y := r.ws(m.Pos)
		bw := math.Max(18, m.Radius*2.2*z)
		u.bar(x-bw/2, y-m.Radius*z-10, bw, 3, m.HP/m.MaxHP, [3]float64{1, 0.3, 0.25}, 0.2)
	}
	if a := g.Selected; a != nil && a.Alive {
		x, y := r.ws(a.Pos)
		rad := (10 + 2*math.Sin(r.time*4)) * math.Max(z, 0.5)
		c := w.Colonies[a.Colony].Color
		u.b.Quad(SprRing, x, y, rad*1.3, rad*1.3, 0, float32(c[0]), float32(c[1]), float32(c[2]), 0.9)
		u.b.Quad(SprGlow, x, y, rad*2, rad*2, 0, float32(c[0]), float32(c[1]), float32(c[2]), 0.2)
	}
}

func (u *UI) header(dst *ebiten.Image, g *Game) {
	w, r := g.World, g.R
	x, y := 18.0, 18.0
	u.panel(dst, r.glassA, x, y, 300, 74, [3]float64{1, 0.85, 0.5})
	u.text(dst, "NeatAnts", u.f.Title, x+16, y+10, [3]float64{1, 0.9, 0.7}, 1, text.AlignStart)

	day := float64(w.Tick)/(sim.Cfg.DayLength*60) + todOffset
	hour := math.Mod(day, 1) * 24
	icon := "Day"
	if r.night > 0.5 {
		icon = "Night"
	}
	weather := "clear sky"
	switch {
	case w.Weird == sim.WeirdFrogs:
		weather = "frog rain"
	case w.Weird == sim.WeirdConfetti:
		weather = "confetti storm"
	case w.Storm:
		weather = "thunderstorm"
	case w.Raining:
		weather = "rain"
	}
	info := fmt.Sprintf("%s %d · %02dh%02d · %s", icon, int(day)+1, int(hour), int(math.Mod(hour, 1)*60), weather)
	u.text(dst, info, u.f.Small, x+16, y+44, colMuted, 1, text.AlignStart)

	speed := g.speedLabel()
	u.text(dst, speed, u.f.H, x+284, y+14, [3]float64{1, 0.85, 0.55}, 1, text.AlignEnd)
	cin := ""
	if g.Cinematic {
		cin = "cinematic camera"
	}
	if sim.FullBrain() {
		cin = "brain only · " + cin
	}
	if s := w.Shaping(); s > sim.Cfg.ShapingFloor && sim.Cfg.ShapingFade > 0 {
		cin = fmt.Sprintf("learning %.0f %% · ", 100*s) + cin
	}
	u.text(dst, cin, u.f.Tiny, x+284, y+48, [3]float64{0.7, 0.8, 1}, 0.8, text.AlignEnd)

	// Sun/moon dial
	cx, cy := x+250.0, y+54.0
	_ = cx
	_ = cy
}

func (u *UI) colonies(dst *ebiten.Image, g *Game) {
	w, r := g.World, g.R
	x, y := 18.0, 104.0
	cols := append([]*sim.Colony{}, w.Colonies...)
	sort.SliceStable(cols, func(i, j int) bool {
		if cols[i].Alive != cols[j].Alive {
			return cols[i].Alive
		}
		return cols[i].Pop > cols[j].Pop
	})
	for _, c := range cols {
		h := 92.0
		spouse := -1
		if c.Alive {
			spouse = w.SpouseOf(c.ID)
		}
		if spouse >= 0 {
			h += 16
		}
		u.panel(dst, r.glassA, x, y, 300, h, c.Color)
		u.b.Line(SprDot, x+7, y+14, x+7, y+h-14, 3, c4(c.Color[0], c.Color[1], c.Color[2], 1))
		u.b.Line(SprGlow, x+7, y+14, x+7, y+h-14, 10, c4(c.Color[0], c.Color[1], c.Color[2], 0.3))
		u.text(dst, c.Name, u.f.H, x+18, y+10, c.Color, 1, text.AlignStart)
		if !c.Alive {
			u.text(dst, fmt.Sprintf("Collapsed · a queen arrives in %ds", max(0, c.DeadTimer/60)), u.f.Small, x+18, y+34, colMuted, 1, text.AlignStart)
			y += h + 8
			continue
		}
		u.text(dst, fmt.Sprintf("%d ants", c.Pop), u.f.Body, x+282, y+11, colText, 1, text.AlignEnd)
		// Food & nest HP
		stock := fmt.Sprintf("Stock %.0f", c.Food)
		if c.Eggs > 0 {
			stock += fmt.Sprintf(" · %d eggs", c.Eggs)
		}
		u.text(dst, stock, u.f.Small, x+18, y+32, colMuted, 1, text.AlignStart)
		u.bar(x+160, y+37, 122, 5, c.HP/sim.Cfg.NestMaxHP, [3]float64{0.95, 0.4, 0.35}, 0.08)
		// Element composition
		bx, bw := x+18, 264.0
		tot := 0
		for _, n := range c.ElemCount {
			tot += n
		}
		if tot > 0 {
			cur := bx
			for e := range sim.NumElements {
				seg := bw * float64(c.ElemCount[e]) / float64(tot)
				if seg > 1 {
					ec := sim.ElemColors[e]
					u.b.Line(SprDot, cur+1, y+57, cur+seg-1, y+57, 6, c4(ec[0], ec[1], ec[2], 0.9))
				}
				cur += seg
			}
		}
		// Stats line
		dom, dn := 0, -1
		for e, n := range c.ElemCount {
			if n > dn {
				dom, dn = e, n
			}
		}
		stats := fmt.Sprintf("Lineage %s · %d deliveries · %d monsters", sim.ElemNames[dom], c.Delivered, c.MonsterKills)
		if c.Cans > 0 {
			stats += fmt.Sprintf(" · %d bombs", c.Cans)
		}
		u.text(dst, stats, u.f.Tiny, x+18, y+68, colMuted, 1, text.AlignStart)
		if spouse >= 0 {
			sp := w.Colonies[spouse]
			u.b.Quad(SprHeart, x+24, y+89, 5, 5, 0, 1, 0.45, 0.65, 1)
			since := (w.Tick - c.MarriedAt) / 3600
			u.text(dst, fmt.Sprintf("Married to %s for %d min", sp.Name, since), u.f.Tiny, x+34, y+82, sim.Mixc([3]float64{1, 0.6, 0.75}, sp.Color, 0.3), 1, text.AlignStart)
		}
		// Population sparkline
		h0 := u.history[c.ID]
		if len(h0) > 3 {
			maxV, minV := 1, 1<<30
			for _, v := range h0 {
				maxV, minV = max(maxV, v), min(minV, v)
			}
			if maxV-minV < 4 {
				minV = maxV - 4
			}
			sx, sy, sw, sh := x+118.0, y+10.0, 86.0, 12.0
			for i := 1; i < len(h0); i++ {
				x0 := sx + sw*float64(i-1)/float64(len(h0)-1)
				x1 := sx + sw*float64(i)/float64(len(h0)-1)
				y0 := sy + sh - sh*float64(h0[i-1]-minV)/float64(maxV-minV)
				y1 := sy + sh - sh*float64(h0[i]-minV)/float64(maxV-minV)
				u.b.Line(SprDot, x0, y0, x1, y1, 1.3, c4(c.Color[0], c.Color[1], c.Color[2], 0.7))
			}
		}
		y += h + 8
	}
}

func (u *UI) wave(dst *ebiten.Image, g *Game) {
	w, r := g.World, g.R
	pw := 380.0
	x, y := SW/2-pw/2, 18.0
	u.panel(dst, r.glassA, x, y, pw, 62, [3]float64{1, 0.4, 0.3})
	title := "The cave sleeps"
	if w.Wave > 0 {
		title = "Wave " + itoa(w.Wave)
	}
	u.text(dst, title, u.f.Title, x+pw/2, y+6, [3]float64{1, 0.55, 0.45}, 1, text.AlignCenter)
	remain := w.NextWave - w.Tick
	if w.WaveAlive > 0 {
		sick := 0
		for _, m := range w.Monsters {
			if m.Alive && m.Fx.Has(sim.FxPlague) {
				sick++
			}
		}
		line := fmt.Sprintf("%d creatures hunting · %d killed", w.WaveAlive, w.TotalKilled)
		if sim.Cfg.AdaptiveWaves {
			line += fmt.Sprintf(" · threat x%.2f", w.Threat)
		}
		if sick > 0 {
			line += fmt.Sprintf(" · %d sick", sick)
		}
		if w.InsectResist > 0.05 {
			line += fmt.Sprintf(" · resistance %.0f %%", w.InsectResist*100)
		}
		u.text(dst, line, u.f.Small, x+pw/2, y+38, [3]float64{1, 0.7, 0.65}, 1, text.AlignCenter)
	} else {
		line := fmt.Sprintf("Next wave in %ds", max(0, remain/60))
		if sim.Cfg.AdaptiveWaves {
			line += fmt.Sprintf(" · threat x%.2f", w.Threat)
		}
		u.text(dst, line, u.f.Small, x+pw/2, y+38, colMuted, 1, text.AlignCenter)
	}
	frac := 1 - float64(remain)/float64(sim.Cfg.WaveInterval)
	if w.Wave == 0 {
		frac = float64(w.Tick) / float64(sim.Cfg.FirstWave)
	}
	u.bar(x+20, y+56, pw-40, 3, frac, [3]float64{1, 0.35, 0.25}, 0.08)

	// Boss bar
	for _, m := range w.Monsters {
		if m.Alive && m.Kind >= sim.MonCentipede {
			by := y + 74
			u.panel(dst, r.glassA, x-60, by, pw+120, 44, [3]float64{1, 0.2, 0.1})
			u.text(dst, sim.MonsterSpecs[m.Kind].Name, u.f.H, x+pw/2, by+6, [3]float64{1, 0.8, 0.5}, 1, text.AlignCenter)
			u.bar(x-40, by+28, pw+80, 8, m.HP/m.MaxHP, [3]float64{0.95, 0.2, 0.12}, 0.12)
			break
		}
	}
}

func (u *UI) chronicle(dst *ebiten.Image, g *Game) {
	w, r := g.World, g.R
	n := min(6, len(w.Log))
	if n == 0 {
		return
	}
	pw, ph := 470.0, 30.0+float64(n)*19
	x, y := 18.0, SH-ph-18
	u.panel(dst, r.glassA, x, y, pw, ph, [3]float64{0.8, 0.8, 0.9})
	u.text(dst, "Chronicles", u.f.H, x+16, y+8, [3]float64{0.95, 0.9, 0.8}, 1, text.AlignStart)
	for i := range n {
		m := w.Log[len(w.Log)-n+i]
		age := float64(w.Tick-m.Tick) / 60
		a := sim.ClampF(1-age/120, 0.35, 1)
		u.b.Circle(x+20, y+37+float64(i)*19, 3, c4(m.Color[0], m.Color[1], m.Color[2], a))
		u.text(dst, m.Text, u.f.Small, x+30, y+30+float64(i)*19, sim.Mixc(colText, m.Color, 0.3), a, text.AlignStart)
	}
}

func (u *UI) minimapDraw(dst *ebiten.Image, g *Game) {
	w, r := g.World, g.R
	u.panel(dst, r.glassA, mmX-8, mmY-8, mmW+16, mmH+16, [3]float64{0.7, 0.8, 1})
	op := &ebiten.DrawImageOptions{}
	op.GeoM.Translate(mmX, mmY)
	op.ColorScale.Scale(0.85, 0.85, 0.85, 1)
	dst.DrawImage(u.minimap, op)
	sx, sy := float64(mmW)/sim.WorldW, float64(mmH)/sim.WorldH
	for _, c := range w.Colonies {
		if !c.Alive {
			continue
		}
		x, y := mmX+c.Pos.X*sx, mmY+c.Pos.Y*sy
		u.b.Glow(x, y, 14, float32(c.Color[0]), float32(c.Color[1]), float32(c.Color[2]), 0.5)
		u.b.Circle(x, y, 4, c4(c.Color[0], c.Color[1], c.Color[2], 1))
	}
	for _, a := range w.Ants {
		if a.Alive {
			c := w.Colonies[a.Colony].Color
			u.b.Circle(mmX+a.Pos.X*sx, mmY+a.Pos.Y*sy, 1, c4(c[0], c[1], c[2], 0.85))
		}
	}
	for _, m := range w.Monsters {
		if m.Alive {
			rad := 1.8
			if m.Kind >= sim.MonCentipede {
				rad = 4
			}
			u.b.Circle(mmX+m.Pos.X*sx, mmY+m.Pos.Y*sy, rad, c4(1, 0.2, 0.15, 1))
		}
	}
	if sim.Cfg.ShopEnabled {
		u.b.Quad(SprSolid, mmX+w.Shop.X*sx, mmY+w.Shop.Y*sy, 4, 3, 0, 0.3, 0.55, 1, 1)
		if k := w.Kiosk; k != nil && k.Alive {
			c := w.Colonies[k.Owner].Color
			u.b.Quad(SprSolid, mmX+k.Pos.X*sx, mmY+k.Pos.Y*sy, 2.5, 2.5, 0.785, float32(c[0]), float32(c[1]), float32(c[2]), 1)
		}
	}
	for _, c := range w.Colonies {
		if s := w.SpouseOf(c.ID); s >= 0 && c.ID < s {
			o := w.Colonies[s]
			x0, y0 := mmX+c.Pos.X*sx, mmY+c.Pos.Y*sy
			x1, y1 := mmX+o.Pos.X*sx, mmY+o.Pos.Y*sy
			u.b.Line(SprGlow, x0, y0, x1, y1, 3, c4(1, 0.45, 0.65, 0.7))
			u.b.Quad(SprHeart, (x0+x1)/2, (y0+y1)/2, 5, 5, 0, 1, 0.45, 0.65, 1)
		}
	}
	for _, f := range w.Frogs {
		u.b.Circle(mmX+f.Pos.X*sx, mmY+f.Pos.Y*sy, 1.2, c4(0.45, 0.95, 0.35, 0.9))
	}
	for _, k := range w.Couriers {
		u.b.Circle(mmX+k.Pos.X*sx, mmY+k.Pos.Y*sy, 1.6, c4(1, 1, 1, 1))
	}
	cx, cy := mmX+w.Cave.X*sx, mmY+w.Cave.Y*sy
	u.b.Glow(cx, cy, 18, 1, 0.2, 0.1, float32(0.4+0.2*math.Sin(r.time*3)))
	u.b.Flush(dst, ebiten.BlendSourceOver)
	// Camera frame
	tl := r.sw(0, 0)
	br := r.sw(SW, SH)
	x0, y0 := mmX+sim.ClampF(tl.X, 0, sim.WorldW)*sx, mmY+sim.ClampF(tl.Y, 0, sim.WorldH)*sy
	x1, y1 := mmX+sim.ClampF(br.X, 0, sim.WorldW)*sx, mmY+sim.ClampF(br.Y, 0, sim.WorldH)*sy
	frame := roundRect(float32(x0), float32(y0), float32(x1-x0), float32(y1-y0), 3)
	strokePath(dst, frame, 1.2, [4]float32{1, 1, 1, 0.7})
}

func (u *UI) drawBanner(dst *ebiten.Image, r *Renderer) {
	b := r.banner
	if b.T <= 0 {
		return
	}
	a := sim.ClampF(math.Min(b.T/1.2, (5-b.T)/0.5), 0, 1)
	y := SH*0.36 - (1-a)*20
	u.b.Glow(SW/2, y+40, 360, float32(b.Col[0]), float32(b.Col[1]), float32(b.Col[2]), float32(0.12*a))
	u.b.Line(SprGlow, SW/2-260*a, y+92, SW/2+260*a, y+92, 4, c4(b.Col[0], b.Col[1], b.Col[2], 0.7*a))
	u.b.Flush(dst, ebiten.BlendLighter)
	u.text(dst, b.Title, u.f.Big, SW/2, y, sim.Mixc(b.Col, [3]float64{1, 1, 1}, 0.4), a, text.AlignCenter)
	u.text(dst, b.Sub, u.f.Body, SW/2, y+102, colText, a*0.9, text.AlignCenter)
}

func (u *UI) help(dst *ebiten.Image, g *Game) {
	r := g.R
	pw, ph := 420.0, 377.0
	x, y := SW/2-pw/2, SH/2-ph/2
	u.panel(dst, r.glassA, x, y, pw, ph, [3]float64{1, 0.9, 0.6})
	u.text(dst, "Controls", u.f.Title, x+pw/2, y+14, [3]float64{1, 0.9, 0.7}, 1, text.AlignCenter)
	rows := [][2]string{
		{"Click", "pick an ant and read its genome"},
		{"Drag, WASD/arrows", "move the camera"},
		{"Wheel", "zoom"},
		{"C", "cinematic camera"},
		{"Space", "pause"},
		{"1 to 5", "speed"},
		{"U", "turbo mode without display"},
		{"P", "pheromones"},
		{"M", "simple mode"},
		{"N", "trigger the next wave"},
		{"Tab", "hide the interface"},
		{"F", "fullscreen"},
		{"Esc", "deselect"},
		{"H", "close this help"},
	}
	for i, row := range rows {
		yy := y + 56 + float64(i)*21
		u.text(dst, row[0], u.f.H, x+150, yy, [3]float64{1, 0.85, 0.55}, 1, text.AlignEnd)
		u.text(dst, row[1], u.f.Body, x+166, yy+1, colText, 1, text.AlignStart)
	}
}

// --- Genome inspector ---

type netLayout struct {
	genomeID int
	pos      map[int][2]float64 // node index -> normalised position
	idx      map[int]int        // node id -> index
}

var outputNames = [sim.AntOutputs]string{"Turn", "Speed", "Trail", "Spell", "Bite", "Alarm", "Fight", "Explore"}

var traitNames = [sim.NumTraits]string{"Fire", "Frost", "Lightning", "Earth", "Life", "Size", "Speed", "Instinct", "Courage", "Curiosity", "Loyalty", "Ardor"}

func (u *UI) inspector(dst *ebiten.Image, g *Game, a *sim.Ant) {
	w, r := g.World, g.R
	pw := 340.0
	x, y := SW-pw-18, 18.0
	ph := SH - mmH - 18 - 18 - 26.0
	col := w.Colonies[a.Colony]
	u.panel(dst, r.glassA, x, y, pw, ph, col.Color)
	ec := sim.ElemColors[a.Element]
	caste := "Worker"
	switch {
	case a.Size > 1.3:
		caste = "Soldier"
	case a.Size < 0.9:
		caste = "Scout"
	}
	status := ""
	if !a.Alive {
		status = " (dead)"
	}
	u.text(dst, fmt.Sprintf("%s #%d%s", caste, a.ID, status), u.f.H, x+16, y+12, colText, 1, text.AlignStart)
	u.text(dst, "Colony "+col.Name, u.f.Small, x+16, y+32, col.Color, 1, text.AlignStart)
	u.b.Glow(x+pw-34, y+28, 26, float32(ec[0]), float32(ec[1]), float32(ec[2]), 0.5)
	u.b.Circle(x+pw-34, y+28, 9, c4(ec[0], ec[1], ec[2], 1))
	u.text(dst, sim.ElemNames[a.Element], u.f.Small, x+pw-52, y+44, ec, 1, text.AlignEnd)
	u.text(dst, fmt.Sprintf("Power %.0f%%", a.Power*100/1.5), u.f.Tiny, x+pw-52, y+14, colMuted, 1, text.AlignEnd)

	yy := y + 60
	stat := func(label string, v float64, c [3]float64) {
		u.text(dst, label, u.f.Tiny, x+16, yy-2, colMuted, 1, text.AlignStart)
		u.bar(x+80, yy+2, pw-100, 6, v, c, 0.1)
		yy += 16
	}
	stat("Health", a.HP/a.MaxHP, [3]float64{0.95, 0.35, 0.35})
	stat("Energy", a.Energy, [3]float64{1, 0.8, 0.35})
	stat("Mana", a.Mana, ec)

	yy += 4
	u.text(dst, fmt.Sprintf("Deliveries %d · Victims %d · Monsters %d · Healing %.0f", a.Delivered, a.Kills, a.MonsterKills, a.Healed), u.f.Tiny, x+16, yy, colText, 0.9, text.AlignStart)
	yy += 16
	u.text(dst, fmt.Sprintf("Age %.0f %% of its life", 100*float64(a.Age)/float64(max(1, a.MaxAge))), u.f.Tiny, x+16, yy, colMuted, 1, text.AlignStart)
	yy += 18
	// Active buffs and debuffs as coloured chips
	cx := x + 16
	chips := 0
	for k := range sim.NumEffects {
		t := a.Fx[k]
		if t <= 0 {
			continue
		}
		info := sim.Effect[k]
		label := info.Name
		if sim.EffectKind(k) != sim.FxOld {
			label = fmt.Sprintf("%s %ds", info.Name, (t+59)/60)
		}
		wd := text.Advance(label, u.f.Tiny) + 14
		if cx+wd > x+pw-16 {
			cx = x + 16
			yy += 18
		}
		chip := roundRect(float32(cx), float32(yy), float32(wd), 15, 7)
		c := info.Color
		fillPath(dst, chip, [4]float32{float32(c[0]) * 0.25, float32(c[1]) * 0.25, float32(c[2]) * 0.25, 0.85})
		strokePath(dst, chip, 1, [4]float32{float32(c[0]), float32(c[1]), float32(c[2]), 0.6})
		u.text(dst, label, u.f.Tiny, cx+7, yy+1, c, 1, text.AlignStart)
		cx += wd + 6
		chips++
	}
	if chips > 0 {
		yy += 22
	} else {
		yy += 4
	}

	// Decision: utility of each role, chosen one highlighted
	decision := "Decision · " + sim.RoleNames[a.Role]
	if sim.FullBrain() {
		decision = "Mood (indicative) · " + sim.RoleNames[a.Role]
	}
	u.text(dst, decision, u.f.H, x+16, yy, [3]float64{1, 0.9, 0.7}, 1, text.AlignStart)
	yy += 22
	maxU := 0.01
	for _, v := range a.Utility {
		maxU = math.Max(maxU, v)
	}
	for ri := range sim.NumRoles {
		c := colMuted
		if sim.Role(ri) == a.Role {
			c = [3]float64{1, 0.85, 0.5}
		}
		cx := x + 16 + float64(ri%2)*158
		cy := yy + float64(ri/2)*15
		u.text(dst, sim.RoleNames[ri], u.f.Tiny, cx, cy-2, c, 1, text.AlignStart)
		u.bar(cx+52, cy+2, 90, 5, a.Utility[ri]/maxU, c, 0.08)
	}
	yy += float64(15*((int(sim.NumRoles)+1)/2) + 6)

	// Genome traits
	u.text(dst, "Genome", u.f.H, x+16, yy, [3]float64{1, 0.9, 0.7}, 1, text.AlignStart)
	yy += 22
	t := a.Genome.Traits
	for i := range sim.NumTraits {
		c := [3]float64{0.75, 0.8, 0.9}
		if i < sim.NumElements {
			c = sim.ElemColors[i]
		}
		cx := x + 16 + float64(i%2)*158
		cy := yy + float64(i/2)*15
		u.text(dst, traitNames[i], u.f.Tiny, cx, cy-2, colMuted, 1, text.AlignStart)
		u.bar(cx+60, cy+2, 82, 5, t[i], c, 0.08)
	}
	yy += 15*6 + 8

	// Neural network
	mem := 0
	for _, n := range a.Genome.Nodes {
		if n.Type == neatMemory {
			mem++
		}
	}
	u.text(dst, fmt.Sprintf("NEAT brain · %d neurons · %d memories · %d synapses", len(a.Genome.Nodes), mem, len(a.Genome.Conns)), u.f.H, x+16, yy, [3]float64{1, 0.9, 0.7}, 1, text.AlignStart)
	yy += 24
	u.network(dst, a, x+16, yy, pw-32, y+ph-yy-12)
}

func (u *UI) network(dst *ebiten.Image, a *sim.Ant, x, y, w, h float64) {
	g := a.Genome
	if u.netCache == nil || u.netCache.genomeID != g.ID {
		u.netCache = layoutNet(g)
	}
	L := u.netCache
	pt := func(i int) (float64, float64) {
		p := L.pos[i]
		return x + p[0]*(w-70), y + p[1]*h
	}
	for _, c := range g.Conns {
		if !c.Enabled {
			continue
		}
		i, ok1 := L.idx[c.In]
		j, ok2 := L.idx[c.Out]
		if !ok1 || !ok2 {
			continue
		}
		x0, y0 := pt(i)
		x1, y1 := pt(j)
		act := math.Abs(a.Net.Value(i))
		al := sim.ClampF(math.Abs(c.Weight)/4, 0.05, 0.6) * (0.3 + 0.7*act)
		cc := [3]float64{1, 0.65, 0.3}
		if c.Weight < 0 {
			cc = [3]float64{0.35, 0.65, 1}
		}
		if c.Gate != 0 {
			cc = [3]float64{0.8, 0.5, 1}
		}
		u.b.Line(SprGlow, x0, y0, x1, y1, 2, c4(cc[0], cc[1], cc[2], al))
	}
	for i, n := range g.Nodes {
		px, py := pt(i)
		v := sim.ClampF(a.Net.Value(i), 0, 1)
		if n.Type == neatSensor || n.Type == neatBias {
			v = sim.ClampF(math.Abs(a.Sense[max(0, min(i-1, sim.AntInputs-1))]), 0, 1)
			if n.Type == neatBias {
				v = 1
			}
		}
		rad := 2.2
		if n.Type == neatOutput {
			rad = 4
		}
		if n.Type == neatMemory {
			// Memory cell: ring whose glow follows the stored state.
			cell := sim.ClampF(math.Abs(a.Net.Cell(i)), 0, 1)
			u.b.Quad(SprRing, px, py, 7, 7, 0, 0.55, 0.9, 1, 0.9)
			u.b.Glow(px, py, 14, 0.45, 0.85, 1, float32(0.15+0.5*cell))
			rad = 3.2
			v = sim.ClampF(math.Abs(a.Net.Value(i)), 0, 1)
		}
		u.b.Circle(px, py, rad+1, c4(0, 0, 0, 0.6))
		u.b.Circle(px, py, rad, c4(0.3+0.7*v, 0.3+0.6*v, 0.35+0.2*v, 1))
		if v > 0.5 {
			u.b.Glow(px, py, rad*3, 1, 0.85, 0.5, float32(v*0.4))
		}
	}
	u.b.Flush(dst, ebiten.BlendSourceOver)
	oi := 0
	for i, n := range g.Nodes {
		if n.Type == neatOutput && oi < sim.AntOutputs {
			px, py := pt(i)
			u.text(dst, outputNames[oi], u.f.Tiny, px+8, py-6, colMuted, 1, text.AlignStart)
			oi++
		}
	}
}
