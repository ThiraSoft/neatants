package render

import "github.com/ThiraSoft/neatants/neat"

const (
	neatSensor = neat.Sensor
	neatHidden = neat.Hidden
	neatOutput = neat.Output
	neatBias   = neat.Bias
	neatMemory = neat.Memory
)

// layoutNet places nodes in columns by their depth in the network.
func layoutNet(g *neat.Genome) *netLayout {
	L := &netLayout{genomeID: g.ID, pos: map[int][2]float64{}, idx: map[int]int{}}
	for i, n := range g.Nodes {
		L.idx[n.ID] = i
	}
	depth := make([]int, len(g.Nodes))
	for range 12 {
		for _, c := range g.Conns {
			if !c.Enabled {
				continue
			}
			i, ok1 := L.idx[c.In]
			j, ok2 := L.idx[c.Out]
			if t := g.Nodes[j].Type; !ok1 || !ok2 || (t != neatHidden && t != neatMemory) {
				continue
			}
			if depth[j] < depth[i]+1 && depth[i] < 10 {
				depth[j] = depth[i] + 1
			}
		}
	}
	maxD := 1
	for i, n := range g.Nodes {
		if n.Type == neatHidden || n.Type == neatMemory {
			maxD = max(maxD, depth[i])
		}
	}
	cols := map[float64][]int{}
	var ins, outs []int
	for i, n := range g.Nodes {
		switch n.Type {
		case neatSensor, neatBias:
			ins = append(ins, i)
		case neatOutput:
			outs = append(outs, i)
		default:
			x := 0.15 + 0.7*float64(max(1, depth[i]))/float64(maxD+1)
			cols[x] = append(cols[x], i)
		}
	}
	spread := func(list []int, x, margin float64) {
		for k, i := range list {
			y := margin + (1-2*margin)*(float64(k)+0.5)/float64(len(list))
			L.pos[i] = [2]float64{x, y}
		}
	}
	spread(ins, 0, 0.01)
	spread(outs, 1, 0.08)
	for x, list := range cols {
		spread(list, x, 0.1)
	}
	return L
}
