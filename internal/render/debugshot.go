package render

import (
	"fmt"
	"github.com/ThiraSoft/neatants/internal/sim"
	"image"
	"image/png"
	"os"
	"strconv"
	"strings"

	"github.com/hajimehoshi/ebiten/v2"
)

// Debug helpers driven by environment variables (screenshots for development).
//
//	NEATANTS_WARM=ticks        simulate ahead before the first frame
//	NEATANTS_SHOTS=60,300      frames at which to save shot_<frame>.png, then quit
//	NEATANTS_TOD=0.9           override time-of-day offset (0 = night, 0.5 = noon)
//	NEATANTS_FORCE=sale,frogs  trigger Black Friday, frog rain, confetti, lives or a marriage (marry)
//	NEATANTS_CAM=x,y,zoom      fixed camera (x=-1 first nest, -2 cave, -3 Wallmart, -4 wedding arch)
type debugShots struct {
	frames []int
	frame  int
}

func newDebugShots() *debugShots {
	d := &debugShots{}
	for _, s := range strings.Split(os.Getenv("NEATANTS_SHOTS"), ",") {
		if n, err := strconv.Atoi(strings.TrimSpace(s)); err == nil {
			d.frames = append(d.frames, n)
		}
	}
	return d
}

func warmup(w *sim.World) {
	if wv, err := strconv.Atoi(os.Getenv("NEATANTS_WAVE")); err == nil {
		w.Wave = wv - 1
		w.NextWave = w.Tick + 30
	}
	force := os.Getenv("NEATANTS_FORCE") // e.g. "sale,frogs,live"
	if strings.Contains(force, "sale") {
		w.NextSale = w.Tick + 1
	}
	if strings.Contains(force, "frogs") {
		w.Weird, w.WeirdUntil = sim.WeirdFrogs, w.Tick+3000
	}
	if strings.Contains(force, "confetti") {
		w.Weird, w.WeirdUntil = sim.WeirdConfetti, w.Tick+3000
	}
	if strings.Contains(force, "marry") && len(w.Colonies) >= 2 {
		w.Marry(w.Colonies[0], w.Colonies[1])
	}
	if strings.Contains(force, "live") {
		for _, c := range w.Colonies {
			c.NextLive = w.Tick + 1
		}
	}
	n, _ := strconv.Atoi(os.Getenv("NEATANTS_WARM"))
	for range n {
		w.Update()
		w.Events = w.Events[:0]
	}
}

func (d *debugShots) after(screen *ebiten.Image) {
	if len(d.frames) == 0 {
		return
	}
	d.frame++
	for _, f := range d.frames {
		if f == d.frame {
			b := screen.Bounds()
			img := image.NewRGBA(b)
			screen.ReadPixels(img.Pix)
			out, _ := os.Create(fmt.Sprintf("shot_%d.png", f))
			png.Encode(out, img)
			out.Close()
		}
	}
	if d.frame >= d.frames[len(d.frames)-1] {
		fmt.Printf("[DEBUG] fps=%.1f tps=%.1f\n", ebiten.ActualFPS(), ebiten.ActualTPS())
		os.Exit(0)
	}
}

var todOffset = func() float64 {
	if v, err := strconv.ParseFloat(os.Getenv("NEATANTS_TOD"), 64); err == nil {
		return v
	}
	return 0.33
}()
