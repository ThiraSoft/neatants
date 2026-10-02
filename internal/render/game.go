package render

import (
	"fmt"
	"github.com/ThiraSoft/neatants/internal/sim"
	"math"
	"math/rand"
	"os"

	"github.com/hajimehoshi/ebiten/v2"
	"github.com/hajimehoshi/ebiten/v2/inpututil"
)

var speedSteps = []int{1, 2, 5, 10, 30}

type Game struct {
	World *sim.World
	R     *Renderer
	UI    *UI

	SpeedIdx  int
	Paused    bool
	Cinematic bool
	HideHUD   bool
	ShowHelp  bool
	Selected  *sim.Ant

	idle      float64
	shotTimer float64
	dragging  bool
	dragMoved bool
	dragX     int
	dragY     int
	anchorW   sim.Vec2
	anchorS   [2]float64
	anchorT   int
	turbo     *Turbo
	boss      *sim.Monster
	follow    *sim.Ant
	shots     *debugShots
}

func NewGame() *Game {
	w := sim.NewWorld()
	warmup(w)
	g := &Game{World: w, R: NewRenderer(), Cinematic: true, shots: newDebugShots()}
	g.UI = NewUI(w)
	if os.Getenv("NEATANTS_TURBO") != "" {
		g.startTurbo()
	}
	if cam := os.Getenv("NEATANTS_CAM"); cam != "" {
		var x, y, z float64
		fmt.Sscanf(cam, "%f,%f,%f", &x, &y, &z)
		if x == -3 {
			x, y = w.Shop.X, w.Shop.Y
		}
		if x == -4 { // first wedding arch
			x, y = w.WeddingArch[0].X, w.WeddingArch[0].Y
		}
		if x == -1 && len(w.Colonies) > 0 {
			x, y = w.Colonies[0].Pos.X, w.Colonies[0].Pos.Y
		}
		if x < 0 {
			x, y = w.Cave.X, w.Cave.Y
		}
		g.R.cam = Camera{X: x, Y: y, Z: z, TX: x, TY: y, TZ: z}
		g.Cinematic = false
		if os.Getenv("NEATANTS_SELECT") != "" {
			for _, a := range w.Ants {
				if a.Alive && a.Pos.Sub(sim.V(x, y)).Len() > 150 {
					g.Selected = a
					break
				}
			}
		}
	}
	return g
}

func (g *Game) speedLabel() string {
	switch {
	case g.Paused:
		return "Pause"
	default:
		return "x" + itoa(speedSteps[g.SpeedIdx])
	}
}

func (g *Game) Update() error {
	if g.turbo != nil {
		g.R.time += 1.0 / 60
		if inpututil.IsKeyJustPressed(ebiten.KeyU) || inpututil.IsKeyJustPressed(ebiten.KeyEscape) {
			g.StopTurbo()
		} else if inpututil.IsKeyJustPressed(ebiten.KeyF) {
			ebiten.SetFullscreen(!ebiten.IsFullscreen())
		}
		return nil
	}
	g.input()
	w := g.World
	if g.turbo != nil {
		return nil
	}

	if !g.Paused {
		for range speedSteps[g.SpeedIdx] {
			g.tick()
		}
	}
	if g.Selected != nil && !g.Selected.Alive && w.Tick%120 == 0 {
		g.Selected = nil
	}
	g.camera()
	g.R.Step(w)
	return nil
}

func (g *Game) tick() {
	w := g.World
	if w.Update() {
		sim.SaveWorld(w)
	}
	if w.Tick%90 == 0 {
		for _, c := range w.Colonies {
			if c.Founded != g.UI.histFrom[c.ID] {
				g.UI.histFrom[c.ID] = c.Founded
				g.UI.history[c.ID] = nil
			}
			h := append(g.UI.history[c.ID], c.Pop)
			if len(h) > 80 {
				h = h[1:]
			}
			g.UI.history[c.ID] = h
		}
	}
}

func (g *Game) input() {
	r := g.R
	g.idle += 1.0 / 60
	touched := func() { g.idle = 0 }

	switch {
	case inpututil.IsKeyJustPressed(ebiten.KeySpace):
		g.Paused = !g.Paused
	case inpututil.IsKeyJustPressed(ebiten.KeyU):
		g.startTurbo()
		return
	case inpututil.IsKeyJustPressed(ebiten.KeyP):
		r.ShowPhero = !r.ShowPhero
	case inpututil.IsKeyJustPressed(ebiten.KeyM):
		r.Simple = !r.Simple
	case inpututil.IsKeyJustPressed(ebiten.KeyC):
		g.Cinematic = !g.Cinematic
		g.shotTimer = 0
	case inpututil.IsKeyJustPressed(ebiten.KeyH):
		g.ShowHelp = !g.ShowHelp
	case inpututil.IsKeyJustPressed(ebiten.KeyTab):
		g.HideHUD = !g.HideHUD
	case inpututil.IsKeyJustPressed(ebiten.KeyF):
		ebiten.SetFullscreen(!ebiten.IsFullscreen())
	case inpututil.IsKeyJustPressed(ebiten.KeyN):
		g.World.NextWave = g.World.Tick + 1
	case inpututil.IsKeyJustPressed(ebiten.KeyEscape):
		g.Selected = nil
	}
	for i, k := range []ebiten.Key{ebiten.Key1, ebiten.Key2, ebiten.Key3, ebiten.Key4, ebiten.Key5} {
		if inpututil.IsKeyJustPressed(k) {
			g.SpeedIdx, g.Paused = i, false
		}
	}

	// Keyboard pan (ZQSD and WASD and arrows)
	pan := 12 / r.cam.Z
	move := func(dx, dy float64) {
		r.cam.TX += dx * pan
		r.cam.TY += dy * pan
		g.Selected = nil
		touched()
	}
	if ebiten.IsKeyPressed(ebiten.KeyArrowLeft) || ebiten.IsKeyPressed(ebiten.KeyA) || ebiten.IsKeyPressed(ebiten.KeyQ) {
		move(-1, 0)
	}
	if ebiten.IsKeyPressed(ebiten.KeyArrowRight) || ebiten.IsKeyPressed(ebiten.KeyD) {
		move(1, 0)
	}
	if ebiten.IsKeyPressed(ebiten.KeyArrowUp) || ebiten.IsKeyPressed(ebiten.KeyW) || ebiten.IsKeyPressed(ebiten.KeyZ) {
		move(0, -1)
	}
	if ebiten.IsKeyPressed(ebiten.KeyArrowDown) || ebiten.IsKeyPressed(ebiten.KeyS) {
		move(0, 1)
	}

	mx, my := ebiten.CursorPosition()
	if _, wy := ebiten.Wheel(); wy != 0 {
		f := math.Pow(1.15, wy)
		r.cam.TZ = sim.ClampF(r.cam.TZ*f, 0.3, 3.5)
		g.anchorS = [2]float64{float64(mx), float64(my)}
		g.anchorW = r.sw(float64(mx), float64(my))
		g.anchorT = 30
		touched()
	}

	inMinimap := mx >= mmX && my >= mmY && mx < mmX+mmW && my < mmY+mmH && !g.HideHUD
	if inpututil.IsMouseButtonJustPressed(ebiten.MouseButtonLeft) || inpututil.IsMouseButtonJustPressed(ebiten.MouseButtonRight) {
		g.dragging, g.dragMoved, g.dragX, g.dragY = true, false, mx, my
	}
	if g.dragging && (ebiten.IsMouseButtonPressed(ebiten.MouseButtonLeft) || ebiten.IsMouseButtonPressed(ebiten.MouseButtonRight)) {
		if inMinimap {
			r.cam.TX = float64(mx-mmX) / mmW * sim.WorldW
			r.cam.TY = float64(my-mmY) / mmH * sim.WorldH
			g.dragMoved = true
			touched()
		} else {
			dx, dy := mx-g.dragX, my-g.dragY
			if g.dragMoved || dx*dx+dy*dy > 25 {
				g.dragMoved = true
				r.cam.TX -= float64(dx) / r.cam.Z
				r.cam.TY -= float64(dy) / r.cam.Z
				r.cam.X, r.cam.Y = r.cam.TX, r.cam.TY
				g.dragX, g.dragY = mx, my
				g.Selected = nil
				touched()
			}
		}
	}
	if g.dragging && inpututil.IsMouseButtonJustReleased(ebiten.MouseButtonLeft) {
		g.dragging = false
		if !g.dragMoved && !inMinimap {
			g.pick(r.sw(float64(mx), float64(my)))
			touched()
		}
	}
	if inpututil.IsMouseButtonJustReleased(ebiten.MouseButtonRight) {
		g.dragging = false
	}
}

func (g *Game) pick(p sim.Vec2) {
	best := 16 / math.Min(g.R.cam.Z, 1.5)
	g.Selected = nil
	for _, a := range g.World.Ants {
		if a.Alive {
			if d := a.Pos.Sub(p).Len(); d < best {
				best, g.Selected = d, a
			}
		}
	}
	if g.Selected != nil && g.R.cam.TZ < 1.4 {
		g.R.cam.TZ = 1.6
	}
}

func (g *Game) camera() {
	r := g.R
	w := g.World
	k := 0.12
	if g.Selected != nil && g.Selected.Alive {
		r.cam.TX, r.cam.TY = g.Selected.Pos.X, g.Selected.Pos.Y
		k = 0.08
	} else if g.Cinematic && g.idle > 5 {
		k = 0.012
		g.shotTimer -= 1.0 / 60
		if g.shotTimer <= 0 {
			g.pickShot()
		}
		if g.boss != nil && g.boss.Alive {
			r.cam.TX, r.cam.TY = g.boss.Pos.X, g.boss.Pos.Y
			k = 0.02
		} else if g.follow != nil && g.follow.Alive {
			r.cam.TX, r.cam.TY = g.follow.Pos.X, g.follow.Pos.Y
			k = 0.025
		}
	}
	z0 := r.cam.Z
	r.cam.Z += (r.cam.TZ - r.cam.Z) * math.Min(1, k*1.6+0.05)
	if g.anchorT > 0 {
		g.anchorT--
		r.cam.X = g.anchorW.X - (g.anchorS[0]-SW/2)/r.cam.Z
		r.cam.Y = g.anchorW.Y - (g.anchorS[1]-SH/2)/r.cam.Z
		r.cam.TX, r.cam.TY = r.cam.X, r.cam.Y
	} else {
		r.cam.X += (r.cam.TX - r.cam.X) * k
		r.cam.Y += (r.cam.TY - r.cam.Y) * k
	}
	_ = z0
	r.clampCam()
	_ = w
}

func (g *Game) pickShot() {
	r, w := g.R, g.World
	g.shotTimer = 9
	g.boss, g.follow = nil, nil
	for _, m := range w.Monsters {
		if m.Alive && m.Kind >= sim.MonCentipede && randf() < 0.6 {
			g.boss = m
			r.cam.TZ = 1.0
			return
		}
	}
	// Strongest recent point of interest
	var best *interest
	for i := range r.interest {
		it := &r.interest[i]
		if best == nil || it.Weight*it.T > best.Weight*best.T {
			best = it
		}
	}
	switch {
	case best != nil && randf() < 0.8:
		r.cam.TX, r.cam.TY = best.Pos.X, best.Pos.Y
		r.cam.TZ = 1.1 + randf()*0.7
		if best.Weight >= 8 {
			r.cam.TZ = 0.8
		}
	case randf() < 0.15:
		r.cam.TX, r.cam.TY, r.cam.TZ = sim.WorldW/2, sim.WorldH/2, 0.42
	case randf() < 0.75 && len(w.Ants) > 0:
		// Follow an ant, preferring ones doing something interesting.
		best := w.Ants[int(randf()*float64(len(w.Ants)))]
		for range 30 {
			a := w.Ants[int(randf()*float64(len(w.Ants)))]
			if a.Alive && (a.Role == sim.RoleFight || a.Role == sim.RoleRescue || a.Carrying || a.Role == sim.RoleExplore) {
				best = a
				break
			}
		}
		g.follow = best
		r.cam.TZ = 1.4 + randf()*0.8
	default:
		var alive []*sim.Colony
		for _, c := range w.Colonies {
			if c.Alive {
				alive = append(alive, c)
			}
		}
		if len(alive) > 0 {
			c := alive[int(randf()*float64(len(alive)))]
			r.cam.TX, r.cam.TY = c.Pos.X+rnd(-150, 150), c.Pos.Y+rnd(-100, 100)
			r.cam.TZ = 0.9 + randf()*0.8
		}
	}
}

func (g *Game) Draw(screen *ebiten.Image) {
	if g.turbo != nil {
		g.UI.drawTurbo(screen, g.turbo, g.R.time)
		g.shots.after(screen)
		return
	}
	var img *ebiten.Image
	if g.R.Simple {
		img = g.R.DrawSimple(g.World)
	} else {
		img = g.R.Draw(g.World)
	}
	screen.DrawImage(img, nil)
	g.UI.Draw(screen, g)
	g.shots.after(screen)
}

func (g *Game) Layout(_, _ int) (int, int) {
	return SW, SH
}

func randf() float64 { return rand.Float64() }
