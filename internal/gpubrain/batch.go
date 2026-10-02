package gpubrain

//go:generate glslc -O --target-env=vulkan1.1 -fshader-stage=compute -DMAX_NODES=1024u -DMAX_EDGES=4096u -DMAX_PLASTIC=256u -DMAX_VALUES=1024u activate.comp -o activate.spv

import (
	_ "embed"
	"fmt"
	"math"
	"sync"
	"sync/atomic"
	"unsafe"

	"github.com/ThiraSoft/golem/vk"
	"github.com/ThiraSoft/neatants/neat"
)

//go:embed activate.spv
var activateSPV []byte

const (
	usageArena = vk.UsageStorage | vk.UsageTransferDst | vk.UsageTransferSrc
	slotBytes  = SlotWords * 4
	noSlot     = math.MaxUint32 // request slot the kernel skips
	stageStart = 256            // slots the staging buffer holds at first, and keeps (18 MB of host memory)
	stageQuiet = 1000           // rounds without a burst before a grown staging buffer is given back
)

// Batch evaluates many networks per dispatch. Each living network keeps a
// slot in a device-local arena from its first evaluation until it stops
// being asked for (ttl rounds): its weights learn and its cells remember on
// the card, and only the inputs and the outputs cross the bus each round.
//
// A round is Open, Add for each request (from any goroutines), Start, then
// Finish once the caller has done something else, then Out.
type Batch struct {
	d                   *vk.Device
	pipe                *vk.Pipeline
	set                 *vk.Set
	arena               *vk.Buffer // Local, capacity*SlotWords words
	prev                *vk.Buffer // arena before this round's growth, copied into arena by Start
	prevSize            int
	req, in             *vk.Buffer // Host: slot per request, inputs
	slots               []uint32   // the slots of the round, kept here: reading the host buffer back is slow
	out                 *vk.Buffer // Readback
	staging             *vk.Buffer // Host: packed newborns of this round
	inStride, outStride int
	ttl, capacity       int
	reqCap, stageCap    int

	mu       sync.Mutex
	free     []int32
	owner    []*neat.Network
	lastSeen []int
	round    int
	births   []birth           // newborns of this round: slot and staging index
	lastBorn int               // births of the previous round
	calm     int               // rounds since a round last needed more than stageStart slots
	cpuOut   [][]float32       // per request below reqCap, set for networks too big for a slot
	overflow map[int][]float32 // outputs of the requests past reqCap, under mu
	count    atomic.Int64      // requests asked this round, overflowing ones included
	n        int               // requests dispatched this round

	uploaded, freed, onCPU int
}

type birth struct{ slot, staged int32 }

// New makes an evaluator for networks of at most inStride inputs and
// outStride outputs. ttl is how many calls a network may skip before its
// slot is freed (think_every + 1). capacity is the initial number of slots.
func New(d *vk.Device, inStride, outStride, ttl, capacity int) (*Batch, error) {
	capacity = max(capacity, 1)
	b := &Batch{d: d, inStride: inStride, outStride: outStride, ttl: ttl, reqCap: capacity, stageCap: stageStart}
	var err error
	if b.pipe, err = d.NewPipeline(activateSPV, 4, 12); err != nil {
		return nil, err
	}
	if b.arena, err = d.Local(capacity*slotBytes, usageArena); err != nil {
		b.Close()
		return nil, err
	}
	if b.staging, err = d.Host(stageStart*slotBytes, vk.UsageTransferSrc); err != nil {
		b.Close()
		return nil, err
	}
	if err = b.allocRequests(capacity); err != nil {
		b.Close()
		return nil, err
	}
	b.addSlots(0, capacity)
	return b, nil
}

// addSlots makes slots [from, to) available. They are pushed highest first
// so that pop hands out the lowest one.
func (b *Batch) addSlots(from, to int) {
	b.owner = append(b.owner, make([]*neat.Network, to-from)...)
	b.lastSeen = append(b.lastSeen, make([]int, to-from)...)
	for s := to - 1; s >= from; s-- {
		b.free = append(b.free, int32(s))
	}
	b.capacity = to
}

// allocRequests (re)creates the per-request buffers for n requests and
// rebuilds the set around them. Only called while nothing is in flight.
func (b *Batch) allocRequests(n int) error {
	for _, x := range []**vk.Buffer{&b.req, &b.in, &b.out} {
		if *x != nil {
			(*x).Close()
			*x = nil
		}
	}
	var err error
	if b.req, err = b.d.Host(n*4, vk.UsageStorage); err != nil {
		return err
	}
	if b.in, err = b.d.Host(n*b.inStride*4, vk.UsageStorage); err != nil {
		return err
	}
	if b.out, err = b.d.Readback(n*b.outStride*4, vk.UsageStorage); err != nil {
		return err
	}
	b.reqCap = n
	b.slots = make([]uint32, n)
	b.cpuOut = make([][]float32, n)
	return b.rebuildSet()
}

func (b *Batch) rebuildSet() error {
	if b.set != nil {
		b.set.Close()
		b.set = nil
	}
	var err error
	b.set, err = b.pipe.NewSet([]*vk.Buffer{b.arena, b.req, b.in, b.out})
	return err
}

// Open starts a round that will take up to about capacity requests, appended
// by Add from any goroutines. Nothing of this batch may be in flight. The
// request buffers grow, by doubling at least, when capacity does not fit or
// when the last round overflowed, so after a warm-up they cover the peak and
// nothing runs on the CPU for lack of room.
func (b *Batch) Open(capacity int) {
	b.round++
	last := int(b.count.Swap(0))
	b.lastBorn = len(b.births)
	b.births = b.births[:0]
	b.shrinkStaging()
	want := max(capacity, b.reqCap)
	if last > b.reqCap {
		want = max(want, 2*b.reqCap, last)
	}
	if want > b.reqCap {
		if err := b.allocRequests(want); err != nil {
			panic("gpubrain: growing the request buffers: " + err.Error())
		}
	}
	clear(b.cpuOut)
	b.overflow = nil
}

// shrinkStaging gives a grown staging buffer back at its starting size once
// no round has needed more than that for stageQuiet rounds. Reallocating it
// sooner would make the next burst of births allocate under the mutex while
// every worker waits. It is not in flight at Open.
func (b *Batch) shrinkStaging() {
	if b.lastBorn > stageStart {
		b.calm = 0
	} else {
		b.calm++
	}
	if b.stageCap == stageStart || b.calm < stageQuiet {
		return
	}
	st, err := b.d.Host(stageStart*slotBytes, vk.UsageTransferSrc)
	if err != nil {
		return // keep the big one
	}
	b.staging.Close()
	b.staging = st
	b.stageCap = stageStart
}

// Add queues net to think about in and returns the request's index, for Out.
// It is safe from many goroutines. A network that is not on the card yet is
// packed into the staging buffer and given a slot. One that does not fit a
// slot, or a request past the round's capacity, is run on the CPU right away,
// so Add never fails and never grows a buffer the card may read. A network
// that stays on the card but overflows once runs on its CPU copy for that
// round, which does not see what the card has learned; it is rare and the
// next Open grows the capacity.
func (b *Batch) Add(net *neat.Network, in []float64) int {
	if len(in) > b.inStride {
		panic(fmt.Sprintf("gpubrain: %d inputs for a batch built for %d", len(in), b.inStride))
	}
	i := int(b.count.Add(1) - 1)
	if i >= b.reqCap {
		out := cpuThink(net, in)
		b.mu.Lock()
		if b.overflow == nil {
			b.overflow = map[int][]float32{}
		}
		b.overflow[i] = out
		b.onCPU++
		b.mu.Unlock()
		return i
	}
	slot := uint32(net.Tag)
	if net.Tag < 0 {
		slot = b.admit(net)
	}
	if slot == noSlot {
		b.cpuOut[i] = cpuThink(net, in)
	} else {
		// Missing inputs read as 0, as on the CPU, so the stale values of
		// the previous round must not stay.
		dst := b.in.Floats()[i*b.inStride : (i+1)*b.inStride]
		for k := range dst {
			dst[k] = 0
			if k < len(in) {
				dst[k] = float32(in[k])
			}
		}
	}
	b.slots[i] = slot
	return i
}

func cpuThink(net *neat.Network, in []float64) []float32 {
	in32 := make([]float32, len(in))
	for k, x := range in {
		in32[k] = float32(x)
	}
	return append([]float32(nil), net.Activate(in32)...)
}

// admit gives a newborn a slot, or returns noSlot when the network is too
// big for one. Everything that touches shared state happens under the mutex.
func (b *Batch) admit(net *neat.Network) uint32 {
	f := net.Flat() // reads net only, so outside the mutex
	big := !fits(f)
	b.mu.Lock()
	defer b.mu.Unlock()
	if big {
		b.onCPU++
		return noSlot
	}
	if len(b.free) == 0 {
		b.growArena()
	}
	slot := b.free[len(b.free)-1]
	b.free = b.free[:len(b.free)-1]
	staged := len(b.births)
	if staged >= b.stageCap {
		b.growStaging()
	}
	pack(b.staging.Uints()[staged*SlotWords:(staged+1)*SlotWords], f)
	b.births = append(b.births, birth{slot, int32(staged)})
	net.Tag = slot
	b.owner[slot] = net
	b.uploaded++
	return uint32(slot)
}

// growArena doubles the arena. It must not record on the device: another
// Batch may have a dispatch in flight on the same one. So the old arena is
// kept and Start records the copy into the new one. Growing twice in a round
// drops the intermediate arena, which never held anything.
func (b *Batch) growArena() {
	oldCap := b.capacity
	arena, err := b.d.Local(2*oldCap*slotBytes, usageArena)
	if err != nil {
		panic("gpubrain: growing the arena: " + err.Error())
	}
	if b.prev == nil {
		b.prev, b.prevSize = b.arena, oldCap*slotBytes
	} else {
		b.arena.Close()
	}
	b.arena = arena
	if err := b.rebuildSet(); err != nil {
		panic("gpubrain: rebuilding the set: " + err.Error())
	}
	b.addSlots(oldCap, 2*oldCap)
}

// growStaging doubles the staging buffer and keeps the newborns already in it.
func (b *Batch) growStaging() {
	st, err := b.d.Host(2*b.stageCap*slotBytes, vk.UsageTransferSrc)
	if err != nil {
		panic("gpubrain: growing the staging buffer: " + err.Error())
	}
	copy(st.Bytes(), b.staging.Bytes())
	b.staging.Close()
	b.staging = st
	b.stageCap *= 2
}

// Start frees the slots nobody asked for in ttl rounds, uploads the newborns
// and dispatches one workgroup per request. It returns at once. Slots are
// freed here, before this round's births are copied, and a freed slot is only
// reusable from the next round, so it can never be handed to a birth that is
// already staged.
func (b *Batch) Start() error {
	b.n = min(int(b.count.Load()), b.reqCap)
	for _, slot := range b.slots[:b.n] {
		if slot != noSlot {
			b.lastSeen[slot] = b.round
		}
	}
	b.mu.Lock()
	for s, net := range b.owner {
		if net != nil && b.round-b.lastSeen[s] > b.ttl {
			net.Tag = -1
			b.owner[s] = nil
			b.free = append(b.free, int32(s))
			b.freed++
		}
	}
	b.mu.Unlock()
	copy(b.req.Uints(), b.slots[:b.n])
	push := [3]uint32{uint32(b.n), uint32(b.inStride), uint32(b.outStride)}
	return b.d.Start(func(r *vk.Recorder) {
		if b.prev != nil {
			r.Copy(b.arena, 0, b.prev, b.prevSize)
			r.Barrier()
		}
		for _, bi := range b.births {
			r.CopyFrom(b.arena, int(bi.slot)*slotBytes, b.staging, int(bi.staged)*slotBytes, slotBytes)
		}
		if len(b.births) > 0 {
			r.Barrier()
		}
		if b.n > 0 {
			r.Dispatch(b.set, uint32(b.n), unsafe.Pointer(&push))
		}
	})
}

// Finish waits for the dispatch Start began.
func (b *Batch) Finish() error {
	err := b.d.Wait()
	if b.prev != nil { // its copy has run
		b.prev.Close()
		b.prev = nil
	}
	return err
}

// Out returns the outputs of request i, valid after Finish. The caller reads
// only as many as its network has. The slice is valid until the next Open.
func (b *Batch) Out(i int) []float32 {
	if i >= b.reqCap {
		return b.overflow[i]
	}
	if b.cpuOut[i] != nil {
		return b.cpuOut[i]
	}
	return b.out.Floats()[i*b.outStride : (i+1)*b.outStride]
}

// Close releases everything the batch holds on the device.
func (b *Batch) Close() {
	if b.set != nil {
		b.set.Close()
	}
	for _, x := range []*vk.Buffer{b.prev, b.arena, b.req, b.in, b.out, b.staging} {
		if x != nil {
			x.Close()
		}
	}
	if b.pipe != nil {
		b.pipe.Close()
	}
}

// Stats returns the networks uploaded, the slots freed and the networks run
// on the CPU since the last call.
func (b *Batch) Stats() (uploaded, freed, cpu int) {
	b.mu.Lock()
	defer b.mu.Unlock()
	uploaded, freed, cpu = b.uploaded, b.freed, b.onCPU
	b.uploaded, b.freed, b.onCPU = 0, 0, 0
	return
}
