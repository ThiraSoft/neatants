package render

import (
	"fmt"
	"github.com/ThiraSoft/neatants/internal/sim"
	"sync"
	"time"

	"github.com/hajimehoshi/ebiten/v2"
	"github.com/hajimehoshi/ebiten/v2/text/v2"
)

// Turbo runs the simulation flat out on its own goroutine while the window
// only shows a summary. Draw never touches the world during turbo: it reads
// a snapshot published under a mutex.
type Turbo struct {
	stop chan struct{}
	done chan struct{}
	mu   sync.Mutex
	snap turboSnap
}

type turboSnap struct {
	Tick, Wave, Killed int
	TPS                float64
	Started            time.Time
	StartTick          int
	Colonies           []colonySnap
	Log                []sim.LogMsg
}

type colonySnap struct {
	Name                         string
	Color                        [3]float64
	Alive                        bool
	Pop, Delivered, MonsterKills int
	Food, Instinct               float64
	Spouse                       string
	Eggs                         int
	ElemCount                    [sim.NumElements]int
}

func (g *Game) startTurbo() {
	t := &Turbo{stop: make(chan struct{}), done: make(chan struct{})}
	t.snap = snapshot(g.World)
	t.snap.Started, t.snap.StartTick = time.Now(), g.World.Tick
	g.turbo = t
	go func() {
		defer close(t.done)
		w := g.World
		last, lastTick := time.Now(), w.Tick
		for {
			select {
			case <-t.stop:
				return
			default:
			}
			g.tick()
			w.Events = w.Events[:0]
			if w.Tick%250 == 0 {
				s := snapshot(w)
				now := time.Now()
				s.TPS = float64(w.Tick-lastTick) / now.Sub(last).Seconds()
				last, lastTick = now, w.Tick
				t.mu.Lock()
				s.Started, s.StartTick = t.snap.Started, t.snap.StartTick
				t.snap = s
				t.mu.Unlock()
			}
		}
	}()
}

// StopTurbo halts the simulation goroutine and waits until it has exited.
func (g *Game) StopTurbo() {
	if g.turbo == nil {
		return
	}
	close(g.turbo.stop)
	<-g.turbo.done
	g.turbo = nil
	g.World.Events = g.World.Events[:0]
	g.Selected, g.follow, g.boss = nil, nil, nil
}

func snapshot(w *sim.World) turboSnap {
	s := turboSnap{Tick: w.Tick, Wave: w.Wave, Killed: w.TotalKilled}
	for _, c := range w.Colonies {
		cs := colonySnap{Name: c.Name, Color: c.Color, Alive: c.Alive, Pop: c.Pop, Delivered: c.Delivered,
			MonsterKills: c.MonsterKills, Food: c.Food, ElemCount: c.ElemCount}
		cs.Eggs = c.Eggs
		if s := w.SpouseOf(c.ID); c.Alive && s >= 0 {
			cs.Spouse = w.Colonies[s].Name
		}
		sum, n := 0.0, 0
		for _, a := range w.Ants {
			if a.Alive && a.Colony == c.ID {
				sum += a.Instinct
				n++
			}
		}
		if n > 0 {
			cs.Instinct = sum / float64(n)
		}
		s.Colonies = append(s.Colonies, cs)
	}
	n := min(6, len(w.Log))
	s.Log = append([]sim.LogMsg{}, w.Log[len(w.Log)-n:]...)
	return s
}

func (u *UI) drawTurbo(dst *ebiten.Image, t *Turbo, time0 float64) {
	t.mu.Lock()
	s := t.snap
	t.mu.Unlock()

	dst.Fill(colorRGB(0.035, 0.04, 0.055))
	u.b.Glow(SW/2, SH*0.42, 700, 1, 0.55, 0.25, 0.05)
	u.b.Flush(dst, ebiten.BlendLighter)

	pw, ph := 760.0, 560.0
	x, y := SW/2-pw/2, SH/2-ph/2
	card := roundRect(float32(x), float32(y), float32(pw), float32(ph), 16)
	fillPath(dst, card, [4]float32{0.07, 0.075, 0.1, 0.95})
	strokePath(dst, card, 1, [4]float32{1, 0.8, 0.5, 0.25})

	u.text(dst, "Turbo mode", u.f.Title, x+32, y+26, [3]float64{1, 0.9, 0.7}, 1, text.AlignStart)
	sub := "The world is no longer drawn, all the power goes to the simulation."
	if sim.FullBrain() {
		sub += " Brain-only mode."
	}
	u.text(dst, sub, u.f.Small, x+32, y+60, colMuted, 1, text.AlignStart)

	elapsed := time.Since(s.Started).Seconds()
	gained := s.Tick - s.StartTick
	u.text(dst, fmt.Sprintf("%.0f", s.TPS), u.f.Big, x+pw-32, y+14, [3]float64{1, 0.8, 0.45}, 1, text.AlignEnd)
	u.text(dst, fmt.Sprintf("ticks/s · x%.0f", s.TPS/60), u.f.Small, x+pw-32, y+86, colMuted, 1, text.AlignEnd)

	day := float64(s.Tick)/(sim.Cfg.DayLength*60) + todOffset
	u.text(dst, fmt.Sprintf("Tick %d · day %d · wave %d · %d monsters killed", s.Tick, int(day)+1, s.Wave, s.Killed),
		u.f.Body, x+32, y+104, colText, 1, text.AlignStart)
	u.text(dst, fmt.Sprintf("%s simulated in %s", fmtDur(float64(gained)/60), fmtDur(elapsed)),
		u.f.Small, x+32, y+126, colMuted, 1, text.AlignStart)

	yy := y + 166
	for _, c := range s.Colonies {
		u.b.Line(SprDot, x+36, yy+4, x+36, yy+30, 3, c4(c.Color[0], c.Color[1], c.Color[2], 1))
		name := c.Name
		if c.Spouse != "" {
			name += " (married to " + c.Spouse + ")"
		}
		u.text(dst, name, u.f.H, x+48, yy, c.Color, 1, text.AlignStart)
		if !c.Alive {
			u.text(dst, "collapsed", u.f.Small, x+48, yy+20, colMuted, 1, text.AlignStart)
			yy += 52
			continue
		}
		u.text(dst, fmt.Sprintf("%d ants · %d eggs · stock %.0f · %d deliveries · %d monsters · instinct %.2f",
			c.Pop, c.Eggs, c.Food, c.Delivered, c.MonsterKills, c.Instinct), u.f.Small, x+48, yy+20, colText, 0.9, text.AlignStart)
		tot := 0
		for _, n := range c.ElemCount {
			tot += n
		}
		if tot > 0 {
			cur, bw := x+pw-232, 200.0
			for e := range sim.NumElements {
				seg := bw * float64(c.ElemCount[e]) / float64(tot)
				if seg > 1 {
					ec := sim.ElemColors[e]
					u.b.Line(SprDot, cur+1, yy+10, cur+seg-1, yy+10, 7, c4(ec[0], ec[1], ec[2], 0.9))
				}
				cur += seg
			}
		}
		yy += 52
	}
	u.b.Flush(dst, ebiten.BlendSourceOver)

	yy = y + ph - 36 - float64(len(s.Log))*18
	for _, m := range s.Log {
		u.text(dst, m.Text, u.f.Tiny, x+32, yy, sim.Mixc(colMuted, m.Color, 0.4), 1, text.AlignStart)
		yy += 18
	}
	pulse := 0.6 + 0.4*sinf(time0*3)
	u.text(dst, "U to go back to the world", u.f.Small, x+pw-32, y+ph-30, [3]float64{1, 0.85, 0.55}, pulse, text.AlignEnd)
}

func fmtDur(sec float64) string {
	switch {
	case sec < 60:
		return fmt.Sprintf("%.0f s", sec)
	case sec < 3600:
		return fmt.Sprintf("%d min %02d s", int(sec)/60, int(sec)%60)
	default:
		return fmt.Sprintf("%d h %02d min", int(sec)/3600, int(sec)%3600/60)
	}
}
