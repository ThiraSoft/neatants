package sim

// cellIndex buckets items (indices into a slice) by grid cell. Each cell is a
// singly linked list; a generation stamp tells whether a cell was filled
// this tick, so rebuilding never touches empty cells: the cost is linear in
// the number of items, not in the size of the grid.
type cellIndex struct {
	head  []int32
	stamp []uint32
	next  []int32
	gen   uint32
}

func newCellIndex(n int) cellIndex {
	return cellIndex{head: make([]int32, n), stamp: make([]uint32, n)}
}

// build indexes n items; cellOfItem returns the cell of item i, or -1.
func (ci *cellIndex) build(n int, cellOfItem func(i int) int32) {
	ci.gen++
	if cap(ci.next) < n {
		ci.next = make([]int32, n, n*2)
	}
	ci.next = ci.next[:n]
	for i := range n {
		c := cellOfItem(i)
		if c < 0 {
			continue
		}
		if ci.stamp[c] != ci.gen {
			ci.stamp[c] = ci.gen
			ci.head[c] = -1
		}
		ci.next[i] = ci.head[c]
		ci.head[c] = int32(i)
	}
}

// first returns the first item of cell c, or -1; iterate with next.
func (ci *cellIndex) first(c int) int32 {
	if ci.stamp[c] != ci.gen {
		return -1
	}
	return ci.head[c]
}

func maxf(a, b float64) float64 {
	if a > b {
		return a
	}
	return b
}

func minf(a, b float64) float64 {
	if a < b {
		return a
	}
	return b
}

// cellCount holds what a sensor cell contains this tick. The stamp tells
// whether the counts are current, so the grid never needs clearing.
type cellCount struct {
	ants  [MaxColonies]uint8
	total uint16
	mon   uint8
	food  uint8
	stamp uint32
}

// touch returns cell c's counts, resetting them on first use this tick.
func (w *World) touch(c int) *cellCount {
	cc := &w.counts[c]
	if cc.stamp != w.countGen {
		*cc = cellCount{stamp: w.countGen}
	}
	return cc
}
