// Command neatants runs the ant-colony game in a window.
//
//	go build -o neatants ./cmd/neatants
//	./neatants                      # reads config.yml from the working directory
//	./neatants -config my.yml
package main

import (
	"flag"
	"log"
	"runtime/debug"

	"github.com/ThiraSoft/neatants/internal/render"
	"github.com/ThiraSoft/neatants/internal/sim"
	"github.com/hajimehoshi/ebiten/v2"
)

func main() {
	config := flag.String("config", "config.yml", "configuration file")
	flag.Parse()

	// The simulation allocates little per tick: fewer, larger GC cycles.
	debug.SetGCPercent(400)
	sim.LoadConfig(*config)
	ebiten.SetWindowSize(render.SW, render.SH)
	ebiten.SetWindowTitle("NeatAnts")
	ebiten.SetWindowResizingMode(ebiten.WindowResizingModeEnabled)
	ebiten.SetTPS(60)
	ebiten.SetVsyncEnabled(true)

	game := render.NewGame()
	err := ebiten.RunGame(game)
	game.StopTurbo()
	sim.SaveWorld(game.World)
	if err != nil {
		log.Fatal(err)
	}
}
