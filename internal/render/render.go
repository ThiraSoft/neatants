package render

import (
	"github.com/ThiraSoft/neatants/internal/sim"
	"math"
	"math/rand"

	"github.com/hajimehoshi/ebiten/v2"
)

const (
	SW = 1600
	SH = 900
)

type Camera struct {
	X, Y, Z    float64
	TX, TY, TZ float64
}

type Shock struct {
	Pos    sim.Vec2
	T, Max float64
	R0, R1 float64
	Str    float64
}

type Bolt struct {
	Pts    []sim.Vec2
	Jag    []sim.Vec2
	Life   float64
	Max    float64
	Col    [3]float32
	Width  float64
	Branch bool
}

type Banner struct {
	Title, Sub string
	T          float64
	Col        [3]float64
}

var (
	blendSubtract = ebiten.Blend{
		BlendFactorSourceRGB:        ebiten.BlendFactorOne,
		BlendFactorSourceAlpha:      ebiten.BlendFactorOne,
		BlendFactorDestinationRGB:   ebiten.BlendFactorOne,
		BlendFactorDestinationAlpha: ebiten.BlendFactorOne,
		BlendOperationRGB:           ebiten.BlendOperationReverseSubtract,
		BlendOperationAlpha:         ebiten.BlendOperationReverseSubtract,
	}
)

type Renderer struct {
	scene, light, fx, final *ebiten.Image
	bloomA, bloomB, bloomUp *ebiten.Image
	glassA, glassB          *ebiten.Image
	decals, paths           *ebiten.Image
	phero                   *ebiten.Image
	pheroPix                []byte

	terrainSh, blurSh, compSh *ebiten.Shader

	bushB, sceneB, shadowB, fxB, lightB, decalB, pathB, topB Batch

	parts  Particles
	shocks []Shock
	bolts  []Bolt
	cam    Camera
	time   float64
	frame  int
	shake  float64
	flash  float64
	night  float64
	day    float64
	elev   float64
	wet    float64
	banner Banner

	ShowPhero bool
	Simple    bool
	simpleBG  *ebiten.Image
	simpleAt  sim.Vec2
	interest  []interest
	nextShot  float64
}

type interest struct {
	Pos    sim.Vec2
	Weight float64
	T      float64
}

func mustShader(src string) *ebiten.Shader {
	s, err := ebiten.NewShader([]byte(src))
	if err != nil {
		panic(err)
	}
	return s
}

func NewRenderer() *Renderer {
	buildAtlas()
	r := &Renderer{
		scene:   makeUnmanaged(SW, SH),
		light:   makeUnmanaged(SW, SH),
		fx:      makeUnmanaged(SW, SH),
		final:   makeUnmanaged(SW, SH),
		bloomUp: makeUnmanaged(SW, SH),
		bloomA:  makeUnmanaged(SW/4, SH/4),
		bloomB:  makeUnmanaged(SW/4, SH/4),
		glassA:  makeUnmanaged(SW/4, SH/4),
		glassB:  makeUnmanaged(SW/4, SH/4),
		decals:  makeUnmanaged(sim.WorldW/2, sim.WorldH/2),
		paths:   makeUnmanaged(sim.WorldW/4, sim.WorldH/4),
		phero:   ebiten.NewImage(sim.GW, sim.GH),

		pheroPix:  make([]byte, sim.GW*sim.GH*4),
		terrainSh: mustShader(terrainSrc),
		blurSh:    mustShader(blurSrc),
		compSh:    mustShader(compositeSrc),
		ShowPhero: true,
	}
	r.cam = Camera{X: sim.WorldW / 2, Y: sim.WorldH / 2, Z: 0.42, TX: sim.WorldW / 2, TY: sim.WorldH / 2, TZ: 0.42}
	return r
}

// --- Coordinates ---

func (r *Renderer) ws(p sim.Vec2) (float64, float64) {
	return (p.X-r.cam.X)*r.cam.Z + SW/2, (p.Y-r.cam.Y)*r.cam.Z + SH/2
}

func (r *Renderer) sw(x, y float64) sim.Vec2 {
	return sim.V((x-SW/2)/r.cam.Z+r.cam.X, (y-SH/2)/r.cam.Z+r.cam.Y)
}

func (r *Renderer) visible(p sim.Vec2, margin float64) bool {
	x, y := r.ws(p)
	m := margin * r.cam.Z
	return x > -m && y > -m && x < SW+m && y < SH+m
}

// camMargin lets the camera pan this many screen pixels past the world edge,
// so a nest in a corner is not stuck under the HUD panels.
const camMargin = 340.0

func (r *Renderer) clampCam() {
	hw, hh := SW/2/r.cam.Z, SH/2/r.cam.Z
	m := camMargin / r.cam.Z
	fit := func(v, h, size float64) float64 {
		if 2*h > size+2*m {
			return size / 2
		}
		return sim.ClampF(v, h-m, size-h+m)
	}
	r.cam.X = fit(r.cam.X, hw, sim.WorldW)
	r.cam.Y = fit(r.cam.Y, hh, sim.WorldH)
	r.cam.TX = fit(r.cam.TX, hw, sim.WorldW)
	r.cam.TY = fit(r.cam.TY, hh, sim.WorldH)
}

// --- Per-frame simulation of visuals ---

func (r *Renderer) Step(w *sim.World) {
	r.time += 1.0 / 60
	r.frame++

	tod := math.Mod(float64(w.Tick)/(sim.Cfg.DayLength*60)+todOffset, 1)
	r.elev = -math.Cos(tod * 2 * math.Pi)
	r.day = smooth(-0.45, 0.1, r.elev)
	r.night = 1 - r.day
	wetT := sim.BoolF(w.Raining)
	r.wet += (wetT - r.wet) * 0.004

	for _, e := range w.Events {
		r.onEvent(w, e)
	}
	w.Events = w.Events[:0]

	r.ambient(w)
	r.weirdAmbient(w)
	r.parts.Step(func(p *Particle) {
		r.decal(int(p.Decal), sim.V(p.X, p.Y), p.Size*1.2, p.Rot, [4]float32{p.R * 0.4, p.G * 0.4, p.B * 0.4, 0.6})
	})

	live := r.shocks[:0]
	for _, s := range r.shocks {
		s.T++
		if s.T < s.Max {
			live = append(live, s)
		}
	}
	r.shocks = live

	bl := r.bolts[:0]
	for _, b := range r.bolts {
		b.Life--
		if b.Life > 0 {
			bl = append(bl, b)
		}
	}
	r.bolts = bl

	r.shake *= 0.88
	r.flash *= 0.75
	if r.banner.T > 0 {
		r.banner.T -= 1.0 / 60
	}
	il := r.interest[:0]
	for _, it := range r.interest {
		it.T -= 1.0 / 60
		if it.T > 0 {
			il = append(il, it)
		}
	}
	r.interest = il
}

func smooth(a, b, x float64) float64 {
	t := sim.ClampF((x-a)/(b-a), 0, 1)
	return t * t * (3 - 2*t)
}

func (r *Renderer) addShock(p sim.Vec2, r1, str, dur float64) {
	if len(r.shocks) < 16 && r.visible(p, 300) {
		r.shocks = append(r.shocks, Shock{Pos: p, Max: dur, R0: 4, R1: r1, Str: str})
	}
}

func (r *Renderer) addShake(p sim.Vec2, amt float64) {
	if r.visible(p, 200) {
		r.shake = math.Min(r.shake+amt, 14)
	}
}

func (r *Renderer) note(p sim.Vec2, weight float64) {
	if len(r.interest) < 64 {
		r.interest = append(r.interest, interest{p, weight, 6})
	}
}

// decal stamps a persistent mark on the ground layer (world/2 resolution).
func (r *Renderer) decal(spr int, p sim.Vec2, size, rot float64, c [4]float32) {
	r.decalB.Quad(spr, p.X/2, p.Y/2, size/2, size/2, rot, c[0], c[1], c[2], c[3])
}

// --- Ambient life ---

func (r *Renderer) ambient(w *sim.World) {
	view := func() sim.Vec2 {
		return r.sw(rnd(-50, SW+50), rnd(-50, SH+50))
	}
	// Fireflies at night, pollen by day
	if r.night > 0.3 && rand.Float64() < r.night*0.28 {
		p := view()
		if !w.InPond(p, 0) {
			r.parts.Add(Particle{X: p.X, Y: p.Y, Z: rnd(4, 20), VX: rnd(-0.3, 0.3), VY: rnd(-0.3, 0.3), VZ: rnd(-0.05, 0.05),
				Max: rnd(200, 400), Size: rnd(1.6, 2.6), R: 0.75, G: 1, B: 0.35, A: 0.9, Spr: SprGlow, Add: true, Light: 0.35, Flicker: true})
		}
	}
	if r.day > 0.5 && rand.Float64() < 0.35 {
		p := view()
		r.parts.Add(Particle{X: p.X, Y: p.Y, Z: rnd(10, 40), VX: rnd(0.1, 0.5), VY: rnd(-0.1, 0.2), Max: rnd(200, 400),
			Size: rnd(0.8, 1.5), R: 1, G: 0.97, B: 0.8, A: 0.5, Spr: SprSoft})
	}
	// Cave embers and dark mist
	if r.visible(w.Cave, 400) {
		for range 2 {
			p := w.Cave.Add(sim.Polar(rnd(0, 6.28), rnd(0, 140)))
			r.parts.Add(Particle{X: p.X, Y: p.Y, VX: rnd(-0.3, 0.3), VY: rnd(-0.3, 0.3), VZ: rnd(0.3, 1.1), Max: rnd(60, 140),
				Size: rnd(1, 2.4), R: 1, G: 0.55, B: 0.15, R2: 0.8, G2: 0.1, B2: 0.02, A: 1, Spr: SprGlow, Add: true, Light: 0.2, Flicker: true})
		}
		if r.frame%3 == 0 {
			p := w.Cave.Add(sim.Polar(rnd(0, 6.28), rnd(0, 80)))
			r.parts.Add(Particle{X: p.X, Y: p.Y, VX: rnd(-0.4, 0.4), VY: rnd(-0.4, 0.4), Max: 160, Size: rnd(20, 40), Grow: 0.25,
				R: 0.12, G: 0.05, B: 0.12, A: 0.35, Spr: SprSmoke, Rot: rnd(0, 6), VRot: rnd(-0.01, 0.01)})
		}
	}
	// Rain splashes
	if r.wet > 0.05 {
		n := int(r.wet * 8)
		for range n {
			p := view()
			r.parts.Add(Particle{X: p.X, Y: p.Y, Max: 18, Size: 2, Grow: 0.35, R: 0.8, G: 0.9, B: 1, A: 0.35 * float32(r.wet), Spr: SprRing})
		}
	}
	// Elemental motes from ants whose mana is ready
	for _, a := range w.Ants {
		if !a.Alive || a.Mana < 0.55 || rand.Float64() > 0.03 || !r.visible(a.Pos, 30) {
			continue
		}
		c := sim.ElemColors[a.Element]
		r.parts.Add(Particle{X: a.Pos.X + rnd(-3, 3), Y: a.Pos.Y + rnd(-3, 3), Z: 2, VZ: rnd(0.2, 0.5), VX: rnd(-0.2, 0.2), VY: rnd(-0.2, 0.2),
			Max: 50, Size: rnd(0.8, 1.6), R: float32(c[0]), G: float32(c[1]), B: float32(c[2]), A: 0.9, Spr: SprGlow, Add: true, Light: 0.1})
	}
	// Fire trails, ember wakes
	for _, b := range w.Balls {
		for range 3 {
			r.parts.Add(Particle{X: b.Pos.X + rnd(-3, 3), Y: b.Pos.Y + rnd(-3, 3), VX: -b.Vel.X*0.15 + rnd(-0.5, 0.5), VY: -b.Vel.Y*0.15 + rnd(-0.5, 0.5),
				VZ: rnd(0, 0.4), Max: rnd(18, 34), Size: rnd(1.5, 4), Grow: -0.06, R: 1, G: 0.8, B: 0.3, R2: 0.9, G2: 0.15, B2: 0.02, A: 1,
				Spr: SprGlow, Add: true, Light: 0.15})
		}
		if r.frame%2 == 0 {
			r.parts.Add(Particle{X: b.Pos.X, Y: b.Pos.Y, VZ: 0.3, Max: 50, Size: 4, Grow: 0.2, R: 0.2, G: 0.18, B: 0.17, A: 0.35,
				Spr: SprSmoke, Rot: rnd(0, 6), VRot: rnd(-0.03, 0.03)})
		}
	}
	// Buffs and debuffs
	for _, m := range w.Monsters {
		if m.Alive && r.visible(m.Pos, 60) {
			r.effectParticles(&m.Fx, m.Pos, m.Radius, 0.4)
		}
	}
	for _, a := range w.Ants {
		if a.Alive && r.visible(a.Pos, 20) {
			r.effectParticles(&a.Fx, a.Pos, 5, 0.06)
		}
	}
}

// --- Events to effects ---

func (r *Renderer) onEvent(w *sim.World, e sim.Event) {
	vis := r.visible(e.Pos, 250)
	r.onWTFEvent(w, e, vis)
	cc := [3]float32{1, 1, 1}
	if e.Colony >= 0 && e.Colony < len(w.Colonies) {
		c := w.Colonies[e.Colony].Color
		cc = [3]float32{float32(c[0]), float32(c[1]), float32(c[2])}
	}
	p := e.Pos
	switch e.Kind {
	case sim.EvDeliver:
		if vis {
			r.parts.Add(Particle{X: p.X, Y: p.Y, Max: 40, Size: 20, Grow: 1.6, R: cc[0], G: cc[1], B: cc[2], A: 0.35, Spr: SprRing, Add: true, Light: 0.2})
		}
	case sim.EvHatch:
		if vis {
			for range 4 {
				r.parts.Add(Particle{X: p.X + rnd(-6, 6), Y: p.Y + rnd(-6, 6), VZ: rnd(0.3, 0.8), Max: 40, Size: rnd(1, 2),
					R: cc[0], G: cc[1], B: cc[2], A: 1, Spr: SprStar, Add: true, Rot: rnd(0, 1)})
			}
		}
	case sim.EvBite:
		if vis && rand.Float64() < 0.5 {
			r.parts.Add(Particle{X: p.X, Y: p.Y, Max: 8, Size: 3, Grow: 0.4, R: 1, G: 1, B: 0.9, A: 0.8, Spr: SprStar, Add: true, Rot: rnd(0, 3)})
		}
	case sim.EvMonsterAttack:
		if vis {
			r.parts.Add(Particle{X: p.X, Y: p.Y, Max: 10, Size: 6, Grow: 0.8, R: 1, G: 0.3, B: 0.2, A: 0.8, Spr: SprStar, Add: true, Rot: rnd(0, 3)})
			for range 4 {
				r.parts.Add(Particle{X: p.X, Y: p.Y, VX: rnd(-1.5, 1.5), VY: rnd(-1.5, 1.5), Max: 25, Size: rnd(3, 6), Grow: 0.2,
					R: 0.45, G: 0.38, B: 0.28, A: 0.4, Spr: SprSmoke, Drag: 0.08})
			}
		}
	case sim.EvAntDeath:
		if vis {
			r.parts.Add(Particle{X: p.X, Y: p.Y, Z: 2, VZ: 0.45, VX: rnd(-0.1, 0.1), Max: 110, Size: 3.2, Grow: -0.015,
				R: cc[0], G: cc[1], B: cc[2], A: 0.9, Spr: SprGlow, Add: true, Light: 0.25})
			for range 6 {
				a := rnd(0, 6.28)
				r.parts.Add(Particle{X: p.X, Y: p.Y, Z: 1, VX: math.Cos(a) * rnd(0.5, 1.5), VY: math.Sin(a) * rnd(0.5, 1.5), VZ: rnd(0.5, 1.5), Grav: 0.08,
					Max: 50, Size: rnd(0.8, 1.4), R: 0.15, G: 0.1, B: 0.08, A: 1, Spr: SprDot, Drag: 0.03})
			}
			r.decal(SprSplat, p, 6, rnd(0, 6), [4]float32{0.06, 0.05, 0.04, 0.4})
		}
	case sim.EvMonsterDeath:
		r.monsterDeath(w, p, sim.MonsterKind(e.Val), vis)
	case sim.EvFireCast:
		if vis {
			r.parts.Add(Particle{X: p.X, Y: p.Y, Max: 12, Size: 10, Grow: 1, R: 1, G: 0.6, B: 0.2, A: 0.8, Spr: SprGlow, Add: true, Light: 0.8})
		}
		r.note(p, 1)
	case sim.EvExplosion:
		r.explosion(p, e.Val, vis)
		r.note(p, 1.5)
	case sim.EvFrostNova:
		r.frostNova(p, e.Val, vis)
		r.note(p, 1.5)
	case sim.EvChain:
		r.chain(e.Pts, [3]float32{0.75, 0.65, 1})
		r.note(p, 1.5)
	case sim.EvQuake:
		r.quake(p, e.Val, e.Elem, vis)
		r.note(p, 1.2)
	case sim.EvHeal:
		r.heal(p, e.Val, e.Pts, vis)
		r.note(p, 1)
	case sim.EvNestHit:
		if vis {
			for range 6 {
				a := rnd(0, 6.28)
				r.parts.Add(Particle{X: p.X, Y: p.Y, VX: math.Cos(a) * rnd(1, 3), VY: math.Sin(a) * rnd(1, 3), VZ: rnd(1, 3), Grav: 0.12, Bounce: true,
					Max: 60, Size: rnd(1.5, 3), R: 0.4, G: 0.3, B: 0.2, A: 1, Spr: SprDot, Drag: 0.02})
			}
			r.addShake(p, 2)
		}
		r.note(p, 3)
	case sim.EvWave:
		r.flash = 0.25
		r.shake = 10
		r.addShock(p, 700, 30, 80)
		r.banner = Banner{Title: "Wave " + itoa(int(e.Val)), Sub: waveSub(int(e.Val)), T: 5, Col: [3]float64{1, 0.35, 0.25}}
		for range 80 {
			a := rnd(0, 6.28)
			sp := rnd(2, 7)
			r.parts.Add(Particle{X: p.X, Y: p.Y, VX: math.Cos(a) * sp, VY: math.Sin(a) * sp, Drag: 0.04, Max: rnd(40, 90), Size: rnd(2, 5),
				R: 1, G: 0.4, B: 0.15, R2: 0.5, G2: 0.05, B2: 0.1, A: 1, Spr: SprGlow, Add: true, Stretch: 1.5, Light: 0.3})
		}
		r.note(p, 10)
	case sim.EvMonsterSpawn:
		if vis {
			for range 12 {
				a := rnd(0, 6.28)
				r.parts.Add(Particle{X: p.X, Y: p.Y, VX: math.Cos(a) * rnd(0.5, 2), VY: math.Sin(a) * rnd(0.5, 2), Drag: 0.03, Max: 70, Size: rnd(10, 22), Grow: 0.3,
					R: 0.18, G: 0.04, B: 0.2, A: 0.5, Spr: SprSmoke, Rot: rnd(0, 6)})
			}
			r.parts.Add(Particle{X: p.X, Y: p.Y, Max: 30, Size: 10, Grow: 3, R: 0.9, G: 0.15, B: 0.4, A: 0.6, Spr: SprRing, Add: true, Light: 0.6})
		}
		if sim.MonsterKind(e.Val) >= sim.MonCentipede {
			r.addShock(p, 400, 25, 60)
			r.shake = 12
			r.note(p, 12)
		}
	case sim.EvColonyFall:
		r.addShock(p, 500, 30, 70)
		r.addShake(p, 12)
		for range 50 {
			a := rnd(0, 6.28)
			r.parts.Add(Particle{X: p.X, Y: p.Y, VX: math.Cos(a) * rnd(0.5, 3), VY: math.Sin(a) * rnd(0.5, 3), Drag: 0.03, Max: rnd(80, 160), Size: rnd(15, 35), Grow: 0.3,
				R: 0.35, G: 0.3, B: 0.25, A: 0.5, Spr: SprSmoke, Rot: rnd(0, 6)})
		}
		for range 30 {
			r.parts.Add(Particle{X: p.X + rnd(-40, 40), Y: p.Y + rnd(-40, 40), VZ: rnd(0.4, 1.2), Max: 150, Size: 3, R: cc[0], G: cc[1], B: cc[2], A: 1, Spr: SprGlow, Add: true, Light: 0.4})
		}
		r.decal(SprScorch, p, 160, 0, [4]float32{1, 1, 1, 0.8})
		r.banner = Banner{Title: w.Colonies[e.Colony].Name + " has fallen", Sub: "A lineage dies out", T: 4, Col: w.Colonies[e.Colony].Color}
		r.note(p, 8)
	case sim.EvColonyFound:
		for range 60 {
			a := rnd(0, 6.28)
			r.parts.Add(Particle{X: p.X + math.Cos(a)*rnd(0, 60), Y: p.Y + math.Sin(a)*rnd(0, 60), VZ: rnd(0.5, 2.5), Max: rnd(60, 140), Size: rnd(1, 3),
				R: cc[0], G: cc[1], B: cc[2], A: 1, Spr: SprStar, Add: true, Rot: rnd(0, 1), Light: 0.2})
		}
		r.parts.Add(Particle{X: p.X, Y: p.Y, Max: 90, Size: 20, Grow: 3, R: cc[0], G: cc[1], B: cc[2], A: 0.8, Spr: SprRune, Add: true, VRot: 0.03, Light: 1})
		r.note(p, 4)
	case sim.EvStrike:
		r.skyStrike(p)
	case sim.EvCaveMove:
		r.caveMove(e.Pts[0], p)
	case sim.EvOldAge:
		if vis {
			r.oldAgeDeath(p, cc, e.Val == 1)
		}
	case sim.EvMonsterOld:
		if vis {
			r.monsterOldDeath(p, sim.MonsterKind(e.Val))
		}
	case sim.EvContagion:
		if vis {
			r.contagion(e.Pts[0], p)
		}
	case sim.EvSpray:
		if vis {
			r.sprayCloud(p, e.Val)
		}
		r.note(p, 1.5)
	case sim.EvShopBuy:
		if vis {
			r.shopBuy(p)
		}
	}
}

func waveSub(n int) string {
	switch {
	case n%sim.Cfg.BossEvery == 0:
		return "The ancestral Centipede awakens"
	case n >= 4 && n%sim.Cfg.BossEvery == 3:
		return "A Magma Golem rumbles in the depths"
	case n == 1:
		return "Something stirs at the back of the cave"
	default:
		return "The cave creatures surge out"
	}
}

func (r *Renderer) explosion(p sim.Vec2, rad float64, vis bool) {
	if !vis {
		return
	}
	r.addShock(p, rad*2.4, 14, 30)
	r.addShake(p, 3)
	r.parts.Add(Particle{X: p.X, Y: p.Y, Max: 10, Size: rad * 1.6, Grow: 2, R: 1, G: 0.9, B: 0.6, A: 1, Spr: SprGlow, Add: true, Light: 2.5})
	r.parts.Add(Particle{X: p.X, Y: p.Y, Max: 22, Size: rad * 0.3, Grow: rad * 0.09, R: 1, G: 0.55, B: 0.15, A: 0.9, Spr: SprRing, Add: true})
	for range 34 {
		a := rnd(0, 6.28)
		sp := rnd(0.8, 3.5)
		r.parts.Add(Particle{X: p.X, Y: p.Y, VX: math.Cos(a) * sp, VY: math.Sin(a) * sp, VZ: rnd(0, 1), Drag: 0.07, Max: rnd(20, 45),
			Size: rnd(5, 12), Grow: -0.1, R: 1, G: 0.85, B: 0.4, R2: 0.9, G2: 0.15, B2: 0.02, A: 1, Spr: SprGlow, Add: true, Light: 0.12})
	}
	for range 18 {
		a := rnd(0, 6.28)
		sp := rnd(3, 7)
		r.parts.Add(Particle{X: p.X, Y: p.Y, VX: math.Cos(a) * sp, VY: math.Sin(a) * sp, VZ: rnd(1, 3), Grav: 0.1, Drag: 0.03, Max: rnd(25, 50),
			Size: rnd(1, 1.8), R: 1, G: 0.9, B: 0.5, R2: 1, G2: 0.3, B2: 0, A: 1, Spr: SprGlow, Add: true, Stretch: 2.5, Bounce: true})
	}
	for range 10 {
		a := rnd(0, 6.28)
		r.parts.Add(Particle{X: p.X + math.Cos(a)*rnd(0, 10), Y: p.Y + math.Sin(a)*rnd(0, 10), VX: math.Cos(a) * rnd(0.3, 1), VY: math.Sin(a) * rnd(0.3, 1),
			VZ: 0.3, Max: rnd(70, 120), Size: rnd(10, 18), Grow: 0.25, R: 0.16, G: 0.14, B: 0.13, A: 0.45, Spr: SprSmoke, Rot: rnd(0, 6), VRot: rnd(-0.02, 0.02)})
	}
	r.decal(SprScorch, p, rad*0.9, rnd(0, 6), [4]float32{1, 1, 1, 0.45})
}

func (r *Renderer) frostNova(p sim.Vec2, rad float64, vis bool) {
	if !vis {
		return
	}
	r.addShock(p, rad*1.8, 9, 26)
	r.parts.Add(Particle{X: p.X, Y: p.Y, Max: 24, Size: 6, Grow: rad / 22, R: 0.6, G: 0.9, B: 1, A: 1, Spr: SprRing, Add: true, Light: 1})
	r.parts.Add(Particle{X: p.X, Y: p.Y, Max: 30, Size: rad, Grow: -0.5, R: 0.5, G: 0.8, B: 1, A: 0.35, Spr: SprGlow, Add: true, Light: 1.2})
	for range 26 {
		a := rnd(0, 6.28)
		sp := rnd(1.5, 4)
		r.parts.Add(Particle{X: p.X, Y: p.Y, Z: 3, VX: math.Cos(a) * sp, VY: math.Sin(a) * sp, VZ: rnd(0.5, 2), Grav: 0.08, Drag: 0.05,
			Max: rnd(40, 70), Size: rnd(2.5, 5), Rot: a, R: 0.8, G: 0.95, B: 1, A: 0.9, Spr: SprShard, Bounce: true})
	}
	for range 30 {
		q := p.Add(sim.Polar(rnd(0, 6.28), rnd(0, rad)))
		r.parts.Add(Particle{X: q.X, Y: q.Y, VZ: rnd(0.05, 0.3), Max: rnd(40, 80), Size: rnd(1.5, 3.5), R: 0.8, G: 0.95, B: 1, A: 1, Spr: SprStar, Add: true, Rot: rnd(0, 1)})
	}
	r.decal(SprFrost, p, rad*1.2, rnd(0, 6), [4]float32{0.8, 0.95, 1, 0.55})
}

func (r *Renderer) chain(pts []sim.Vec2, col [3]float32) {
	if len(pts) < 2 {
		return
	}
	r.bolts = append(r.bolts, Bolt{Pts: pts, Life: 16, Max: 16, Col: col, Width: 1})
	for _, q := range pts[1:] {
		if !r.visible(q, 100) {
			continue
		}
		r.parts.Add(Particle{X: q.X, Y: q.Y, Max: 14, Size: 16, Grow: 1, R: col[0], G: col[1], B: col[2], A: 0.8, Spr: SprGlow, Add: true, Light: 1.5})
		for range 8 {
			a := rnd(0, 6.28)
			sp := rnd(2, 5)
			r.parts.Add(Particle{X: q.X, Y: q.Y, VX: math.Cos(a) * sp, VY: math.Sin(a) * sp, VZ: rnd(0.5, 2), Grav: 0.1, Drag: 0.05, Max: rnd(12, 25),
				Size: rnd(0.8, 1.4), R: 0.9, G: 0.9, B: 1, A: 1, Spr: SprGlow, Add: true, Stretch: 2})
		}
		r.decal(SprScorch, q, 10, 0, [4]float32{1, 1, 1, 0.4})
	}
	r.addShock(pts[len(pts)-1], 60, 5, 14)
}

func (r *Renderer) quake(p sim.Vec2, rad float64, elem int, vis bool) {
	if !vis {
		return
	}
	dust := [3]float32{0.62, 0.5, 0.34}
	glow := [3]float32{1, 0.75, 0.35}
	if elem < 0 {
		glow = [3]float32{1, 0.35, 0.05}
	}
	r.addShock(p, rad*2.2, 16, 30)
	r.addShake(p, 4)
	r.parts.Add(Particle{X: p.X, Y: p.Y, Max: 26, Size: 8, Grow: rad / 14, R: glow[0], G: glow[1], B: glow[2], A: 0.8, Spr: SprRing, Add: true, Light: 0.8})
	for range 22 {
		a := rnd(0, 6.28)
		sp := rnd(1, 2.5)
		r.parts.Add(Particle{X: p.X + math.Cos(a)*8, Y: p.Y + math.Sin(a)*8, VX: math.Cos(a) * sp, VY: math.Sin(a) * sp, Drag: 0.05, Max: rnd(50, 90),
			Size: rnd(8, 15), Grow: 0.25, R: dust[0], G: dust[1], B: dust[2], A: 0.45, Spr: SprSmoke, Rot: rnd(0, 6)})
	}
	for range 16 {
		a := rnd(0, 6.28)
		sp := rnd(1, 3)
		r.parts.Add(Particle{X: p.X, Y: p.Y, VX: math.Cos(a) * sp, VY: math.Sin(a) * sp, VZ: rnd(2, 4.5), Grav: 0.16, Bounce: true, Drag: 0.01,
			Max: rnd(60, 100), Size: rnd(1.5, 3), R: 0.42, G: 0.36, B: 0.3, A: 1, Spr: SprDot})
	}
	r.decal(SprCrack, p, rad*1.3, rnd(0, 6), [4]float32{1, 1, 1, 0.8})
}

func (r *Renderer) heal(p sim.Vec2, rad float64, targets []sim.Vec2, vis bool) {
	if !vis {
		return
	}
	r.parts.Add(Particle{X: p.X, Y: p.Y, Max: 40, Size: 8, Grow: rad / 36, R: 0.5, G: 1, B: 0.55, A: 0.7, Spr: SprRing, Add: true, Light: 0.8})
	r.parts.Add(Particle{X: p.X, Y: p.Y, Max: 50, Size: 20, Grow: 0.5, R: 0.6, G: 1, B: 0.5, A: 0.6, Spr: SprRune, Add: true, VRot: 0.05, Light: 0.6})
	for i := range 24 {
		a := float64(i) / 24 * 6.28
		q := p.Add(sim.Polar(a, rnd(5, rad*0.6)))
		r.parts.Add(Particle{X: q.X, Y: q.Y, VX: -math.Sin(a) * 0.6, VY: math.Cos(a) * 0.6, VZ: rnd(0.4, 1.2), Max: rnd(50, 90), Size: rnd(1.2, 2.6),
			R: 0.7, G: 1, B: 0.6, R2: 1, G2: 1, B2: 0.8, A: 1, Spr: SprStar, Add: true, Rot: rnd(0, 1)})
	}
	for range 8 {
		q := p.Add(sim.Polar(rnd(0, 6.28), rnd(0, rad*0.5)))
		r.parts.Add(Particle{X: q.X, Y: q.Y, Z: 10, VX: rnd(-0.5, 0.5), VY: rnd(-0.5, 0.5), VZ: 0.2, Grav: 0.01, Max: rnd(80, 120), Size: rnd(2.5, 4),
			Rot: rnd(0, 6), VRot: rnd(-0.08, 0.08), R: 0.55, G: 0.85, B: 0.35, A: 0.9, Spr: SprLeaf})
	}
	for _, t := range targets {
		d := t.Sub(p)
		for k := range 6 {
			f := float64(k) / 6
			q := p.Add(d.Scale(f))
			r.parts.Add(Particle{X: q.X, Y: q.Y, Z: math.Sin(f*math.Pi) * 14, VX: d.X * 0.01, VY: d.Y * 0.01, Max: 25 + float64(k)*2, Size: 2.5,
				R: 0.6, G: 1, B: 0.6, A: 0.8, Spr: SprGlow, Add: true, Light: 0.1})
		}
	}
}

func (r *Renderer) caveMove(old, p sim.Vec2) {
	// The old mouth caves in under a cloud of dust.
	r.addShock(old, 420, 26, 60)
	for range 45 {
		a := rnd(0, 6.28)
		q := old.Add(sim.Polar(a, rnd(0, 120)))
		r.parts.Add(Particle{X: q.X, Y: q.Y, VX: math.Cos(a) * rnd(0.3, 2), VY: math.Sin(a) * rnd(0.3, 2), Drag: 0.03, Max: rnd(90, 170),
			Size: rnd(18, 40), Grow: 0.3, R: 0.32, G: 0.27, B: 0.24, A: 0.5, Spr: SprSmoke, Rot: rnd(0, 6)})
	}
	r.decal(SprCrack, old, 260, rnd(0, 6), [4]float32{1, 1, 1, 0.9})
	// The new one tears open in a burst of magma.
	r.addShock(p, 600, 32, 70)
	r.addShake(p, 12)
	r.addShake(old, 8)
	for range 90 {
		a := rnd(0, 6.28)
		sp := rnd(1.5, 6)
		r.parts.Add(Particle{X: p.X, Y: p.Y, VX: math.Cos(a) * sp, VY: math.Sin(a) * sp, VZ: rnd(1, 4), Grav: 0.1, Drag: 0.03, Bounce: true,
			Max: rnd(50, 110), Size: rnd(1.5, 4), R: 1, G: 0.7, B: 0.2, R2: 0.9, G2: 0.15, B2: 0.02, A: 1, Spr: SprGlow, Add: true, Light: 0.25, Stretch: 1.2})
	}
	r.parts.Add(Particle{X: p.X, Y: p.Y, Max: 40, Size: 60, Grow: 8, R: 1, G: 0.4, B: 0.1, A: 0.8, Spr: SprRing, Add: true, Light: 2})
	r.note(p, 9)
}

func (r *Renderer) skyStrike(p sim.Vec2) {
	top := p.Add(sim.V(rnd(-200, 200), -900))
	var pts []sim.Vec2
	for i := 0; i <= 6; i++ {
		f := float64(i) / 6
		pts = append(pts, sim.LerpV(top, p, f).Add(sim.V(rnd(-40, 40)*(1-f), 0)))
	}
	pts[len(pts)-1] = p
	r.bolts = append(r.bolts, Bolt{Pts: pts, Life: 18, Max: 18, Col: [3]float32{0.85, 0.85, 1}, Width: 2.5})
	r.flash = 0.3
	if r.visible(p, 300) {
		r.addShock(p, 260, 22, 40)
		r.shake = 8
		r.explosion(p, 30, true)
		r.decal(SprCrack, p, 80, rnd(0, 6), [4]float32{1, 1, 1, 0.9})
	}
	r.note(p, 3)
}

func (r *Renderer) monsterDeath(w *sim.World, p sim.Vec2, k sim.MonsterKind, vis bool) {
	if !vis {
		return
	}
	s := sim.MonsterSpecs[k]
	ichor := [3]float32{0.35, 0.6, 0.15}
	if k == sim.MonGolem {
		ichor = [3]float32{1, 0.4, 0.05}
	}
	big := k >= sim.MonCentipede
	n := 18
	if big {
		n = 70
		r.addShock(p, 400, 30, 60)
		r.shake = 14
		r.flash = 0.3
		r.explosion(p, 60, true)
	}
	r.addShock(p, s.Radius*5, 8, 20)
	for range n {
		a := rnd(0, 6.28)
		sp := rnd(1, 4)
		r.parts.Add(Particle{X: p.X, Y: p.Y, Z: 2, VX: math.Cos(a) * sp, VY: math.Sin(a) * sp, VZ: rnd(1, 3), Grav: 0.12, Drag: 0.02,
			Max: rnd(40, 80), Size: rnd(1.5, 3.5), R: ichor[0], G: ichor[1], B: ichor[2], A: 1, Spr: SprDot, Decal: SprSplat})
	}
	for range n / 2 {
		a := rnd(0, 6.28)
		r.parts.Add(Particle{X: p.X, Y: p.Y, VX: math.Cos(a) * rnd(1, 3), VY: math.Sin(a) * rnd(1, 3), Drag: 0.06, Max: rnd(20, 40), Size: rnd(3, 7),
			R: 1, G: 0.9, B: 0.7, R2: 1, G2: 0.4, B2: 0.2, A: 0.9, Spr: SprGlow, Add: true, Light: 0.2})
	}
	r.decal(SprSplat, p, s.Radius*2, rnd(0, 6), [4]float32{ichor[0] * 0.5, ichor[1] * 0.5, ichor[2] * 0.5, 0.45})
}

func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	neg := n < 0
	if neg {
		n = -n
	}
	var b [20]byte
	i := len(b)
	for n > 0 {
		i--
		b[i] = byte('0' + n%10)
		n /= 10
	}
	if neg {
		i--
		b[i] = '-'
	}
	return string(b[i:])
}
