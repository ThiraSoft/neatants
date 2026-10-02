package sim

import (
	"fmt"
	"github.com/ThiraSoft/neatants/neat"
	"math"
	"math/rand"
	"sync"
)

// Dimensions are compile-time constants because grids are sized from them.
const (
	WorldW = 3600
	WorldH = 2200

	Cell  = 24
	GW    = (WorldW + Cell - 1) / Cell
	GH    = (WorldH + Cell - 1) / Cell
	Cells = GW * GH

	HashCell = 48
	HW       = (WorldW + HashCell - 1) / HashCell
	HH       = (WorldH + HashCell - 1) / HashCell

	MaxColonies = 6
	NestRadius  = 42.0
	// Food picked up closer than this to the nest still feeds it, but the
	// ant gets no delivery for it: turning on the spot is not foraging.
	DeliveryMinDist = NestRadius * 2
	// No food ever appears closer than this to a living nest, so a colony
	// has to walk to eat (the upper bound of the curriculum in foraging.go
	// when food_gap_adaptive is on). Keep it above DeliveryMinDist.
	FoodNestGap = 250.0

	NumSenseDirs = 6
	SenseCh      = 6
	AntStateIn   = 16                                                   // nest, body and clock inputs after the sector rays
	TargetIn     = 9                                                    // nearest food, threat and sister vectors
	AntInputs    = NumSenseDirs*SenseCh + AntStateIn + TargetIn + 3 + 3 // + goal and bush vectors
	AntOutputs   = 8
)

// Heritable traits stored in the genome.
const (
	TraitFire = iota
	TraitFrost
	TraitStorm
	TraitEarth
	TraitLife
	TraitSize
	TraitSpeed
	TraitInstinct
	TraitCourage   // fight rather than flee
	TraitCuriosity // explore far from known paths
	TraitLoyalty   // guard the nest, answer alarms
	TraitDiligence // forage drive
	NumTraits
)

// Role is the high-level behaviour an ant chooses from its genome.
type Role int

const (
	RoleForage Role = iota
	RoleReturn
	RoleFight
	RoleFlee
	RoleGuard
	RoleExplore
	RoleRescue
	RoleRaid
	RoleShop
	RoleWatch // watching the queen's live stream
	NumRoles
)

var RoleNames = [NumRoles]string{"Forages", "Returns", "Fights", "Flees", "Guards", "Explores", "Rescues", "Raids", "Shops", "Watches the live"}

const (
	ElemFire = iota
	ElemFrost
	ElemStorm
	ElemEarth
	ElemLife
	NumElements
)

var ElemNames = [NumElements]string{"Fire", "Frost", "Lightning", "Earth", "Life"}

var ElemColors = [NumElements][3]float64{
	{1.0, 0.45, 0.12},
	{0.45, 0.85, 1.0},
	{0.72, 0.62, 1.0},
	{0.85, 0.66, 0.35},
	{0.45, 1.0, 0.45},
}

type Vec2 struct{ X, Y float64 }

func (v Vec2) Add(o Vec2) Vec2        { return Vec2{v.X + o.X, v.Y + o.Y} }
func (v Vec2) Sub(o Vec2) Vec2        { return Vec2{v.X - o.X, v.Y - o.Y} }
func (v Vec2) Scale(s float64) Vec2   { return Vec2{v.X * s, v.Y * s} }
func (v Vec2) Len() float64           { return math.Sqrt(v.X*v.X + v.Y*v.Y) }
func (v Vec2) Len2() float64          { return v.X*v.X + v.Y*v.Y }
func Polar(a, r float64) Vec2         { return Vec2{math.Cos(a) * r, math.Sin(a) * r} }
func LerpV(a, b Vec2, t float64) Vec2 { return Vec2{a.X + (b.X-a.X)*t, a.Y + (b.Y-a.Y)*t} }

type Ant struct {
	ID       int
	Pos      Vec2
	Prev     Vec2
	Angle    float64
	Heading  float64 // smoothed angle, for rendering
	Gait     float64
	Colony   int
	Genome   *neat.Genome
	Net      *neat.Network
	HP       float64
	MaxHP    float64
	Energy   float64
	Mana     float64
	Carrying bool
	CarryK   uint8
	CarryFar bool // picked up far enough from the nest to count as a delivery

	Element  int
	Power    float64
	Size     float64
	SpeedMul float64
	Instinct float64

	Age, MaxAge      int
	Delivered, Kills int
	MonsterKills     int
	Healed           float64
	Impact           float64 // damage dealt plus healing given, for fitness
	Pickups          int
	Casts, Wasted    int     // spells cast, and those that touched nothing useful
	Eaten            int     // food items eaten on the spot
	HomeProgress     float64 // signed distance walked towards the nest while carrying food
	BornWealth       float64 // colony wealth when she hatched (colony-level fitness)
	BornPop          int
	BornFounded      int
	BiteCD, CastCD   int
	Fx               Effects
	Flash            float64
	LastCast         int
	Alive            bool
	wander           float64
	lastOut          [AntOutputs]float64
	feltHP           float64  // HP at the last thought, for the pain input
	foundAt          int      // Age at the last pickup, meal or delivery
	monsterHitAt     int      // tick of the last monster blow, to blame her death
	hitBy            *Monster // the monster that struck that blow, credited with her death
	Sense            [AntInputs]float64

	Role                               Role
	Courage, Curiosity, Loyalty, Drive float64
	Utility                            [NumRoles]float64
	RaidTarget                         int

	Money, Pack bool // shopping errand: carrying money, carrying the bought pack
	Thief       bool // on the way to shoplift
	Stolen      bool // carrying a stolen pack (the guard is after her)
	ShopAt      int  // SrcWallmart or SrcKiosk
	Paid        float64
	WaitShop    int
	Can         int // insecticide sprays left
	SprayCD     int
}

type Food struct {
	Pos   Vec2
	Kind  uint8
	Taken bool
	Age   int
	Rot   float64
}

type Bush struct {
	Pos  Vec2
	Kind uint8
	R    float64
	Seed float64
}

type Pond struct {
	Pos Vec2
	R   float64
}

type Colony struct {
	ID           int
	Name         string
	Color        [3]float64
	Pos          Vec2
	Food         float64
	HP           float64
	Alive        bool
	Hall, Recent []*neat.Genome
	Species      Speciation
	BiasElem     int
	Born         int
	Delivered    int
	Kills        int
	MonsterKills int
	Deaths       int
	SpawnCD      int
	Pop          int
	ElemCount    [NumElements]int
	DeadTimer    int
	Founded      int
	Pulse        float64
	HitFlash     float64
	monsterHitAt int // tick of the last monster blow on the nest
	raidHitAt    int // tick of the last rival raider blow on the nest
	LastRaidLog  int
	Cans         int // insecticide cans stored at the nest
	LastShopLog  int
	Spouse       int // allied colony (-1 when single)
	Grief        int // wounds received from the spouse
	GiftStreak   int // food gifts given in a row to the spouse
	Abandon      int // nest attacks the spouse did not help with
	MarriedAt    int
	Eggs         int     // brood stored at the nest, hatches even when food runs out
	Wealth       float64 // food produced minus food lost, since founding
	Moves        int     // times the colony had to move its nest
	LastQueenEgg int
	LiveUntil    int // the queen is streaming until this tick
	NextLive     int
	Likes        int
	Generation   int
}

type EventKind int

const (
	EvDeliver EventKind = iota
	EvAntDeath
	EvMonsterDeath
	EvFireCast
	EvExplosion
	EvFrostNova
	EvChain
	EvQuake
	EvHeal
	EvBite
	EvNestHit
	EvWave
	EvMonsterSpawn
	EvColonyFall
	EvColonyFound
	EvStrike
	EvHatch
	EvMonsterAttack
	EvCaveMove
	EvOldAge     // ant dies peacefully (Val 1: starvation)
	EvMonsterOld // monster crumbles with age
	EvContagion  // disease jumps from Pts[0] to Pos
	EvShopBuy    // an ant pays at the Wallmart
	EvSpray      // insecticide cone, Val = direction
	EvSaleStart
	EvStolen
	EvGuardBite
	EvKioskOpen
	EvCourierArrive
	EvCourierHit
	EvWeirdStart // Val = weird weather kind
	EvFrogLand
	EvFrogEat // Pts[0] = frog position
	EvLiveStart
	EvWedding // Pts = both nests, Val = second colony
	EvDivorce
	EvGift     // food shared between spouses, Pts[0] = giver
	EvNestMove // a colony moves after losing its nest, Pts[0] = old nest
)

type Event struct {
	Kind   EventKind
	Pos    Vec2
	Colony int
	Elem   int
	Val    float64
	Pts    []Vec2
}

type LogMsg struct {
	Text  string
	Tick  int
	Color [3]float64
}

type World struct {
	Tick    int
	Evolved int // ticks of evolution of the lineages, kept across saves (fades the shaping rewards)

	// Monster coevolution and adaptive waves (monsterbrain.go)
	MonHall, MonRecent []*neat.Genome
	MonSpecies         Speciation
	Threat             float64
	slain, outlived    int // monsters killed by ants / dead otherwise, since the last wave
	nestsLost          int
	antDeaths          int     // since the last wave, all causes
	antsByMonsters     int     // ... of which killed by monsters (blows or their poison)
	threatDeaths       float64 // ant deaths and those by monsters, smoothed over waves
	threatByMon        float64

	// Foraging curriculum (foraging.go)
	FoodGap               float64
	gapDeaths, gapStarved int
	Ants                  []*Ant
	Monsters              []*Monster
	Foods                 []Food
	Bushes                []Bush
	Ponds                 []Pond
	Colonies              []*Colony
	Balls                 []*Fireball
	Cave                  Vec2
	Shop                  Vec2

	ShopStock  int
	SaleUntil  int
	NextSale   int
	Guard      *Guard
	Kiosk      *Kiosk
	Couriers   []*Courier
	Frogs      []*Frog
	Weird      int
	WeirdUntil int
	NextWeird  int

	InsectResist float64              // monsters' resistance to insecticide, 0..0.85
	AimShots     int                  // aimed spells cast in full-brain mode (fire, storm)
	AimHits      int                  // ... that hit at least one foe
	ElemCharge   [NumElements]float64 // ambient magic of each element, spent by casting

	lastFight       [MaxColonies][MaxColonies]int // last hostile act between two colonies
	WeddingArch     [MaxColonies]Vec2
	lastTheftLog    int
	lastDeliveryLog int

	Trail [MaxColonies][]float32
	Alarm [MaxColonies][]float32

	counts   []cellCount // per sensor cell, valid when stamp == countGen
	countGen uint32
	foodIdx  cellIndex // food per sensor cell

	antHash cellIndex // ants per hash cell
	monHash cellIndex // monsters per hash cell

	Thoughts []Thought // brains to evaluate this tick, between Sense and Act
	thinkers []*Ant    // the ants of Thoughts, in order
	sensed   int       // len(Ants) at Sense: the ants that act this tick
	wg       sync.WaitGroup
	// serialBrains runs every brain on the world's own goroutine: when
	// several worlds evolve side by side, each one already fills a core.
	SerialBrains bool
	shopperCount [MaxColonies]int

	Wave        int
	NextWave    int
	SpawnQueue  []MonsterKind
	spawnCD     int
	WaveTarget  int
	WaveAlive   int
	Clock       float64 // global oscillator shared by every brain, set each tick
	bushFood    []int   // food items lying around each bush
	Learn       LearnStats
	Falls       FallStats
	Raining     bool
	RainTimer   int
	Storm       bool
	LastStrike  int
	Events      []Event
	Log         []LogMsg
	nextID      int
	usedNames   map[string]bool
	TotalKilled int

	lastPlagueLog int
}

func NewWorld() *World {
	w := &World{
		Cave:       Vec2{300 + rand.Float64()*(WorldW-600), 300 + rand.Float64()*(WorldH-600)},
		NextWave:   Cfg.FirstWave,
		RainTimer:  3000 + rand.Intn(6000),
		NextWeird:  int(Cfg.DayLength * 60 * (1 + rand.Float64()*0.6)),
		ElemCharge: [NumElements]float64{1, 1, 1, 1, 1},
		usedNames:  map[string]bool{},
	}
	for c := range MaxColonies {
		w.Trail[c] = make([]float32, Cells)
		w.Alarm[c] = make([]float32, Cells)
	}
	w.counts = make([]cellCount, Cells)
	w.foodIdx = newCellIndex(Cells)
	w.antHash = newCellIndex(HW * HH)
	w.monHash = newCellIndex(HW * HH)

	w.generateTerrain()
	w.placeShop()

	restored := LoadSave()
	saved := restored.Pools
	w.Evolved, w.Threat, w.MonHall = restored.Evolved, restored.Threat, restored.Monsters
	if restored.MonSpecies != nil {
		w.MonSpecies = *restored.MonSpecies
	}
	if w.Threat <= 0 {
		w.Threat = 1
	}
	w.FoodGap = ClampF(restored.FoodGap, DeliveryMinDist, FoodNestGap)
	for i := range min(Cfg.Colonies, MaxColonies) {
		var pool []*neat.Genome
		if i < len(saved) {
			pool = saved[i]
		}
		c := w.foundColony(i, w.freeNestSite(), pool)
		if i < len(restored.Species) && restored.Species[i] != nil {
			c.Species = *restored.Species[i]
		}
	}
	for range Cfg.FoodBushes * 3 {
		w.dropFood(w.Bushes[rand.Intn(len(w.Bushes))])
	}
	w.log("The world awakens. Four queens dig their nests.", [3]float64{0.9, 0.9, 0.8})
	return w
}

func (w *World) generateTerrain() {
	farFromAll := func(p Vec2, r float64) bool {
		if p.Sub(w.Cave).Len() < 380+r {
			return false
		}
		for _, o := range w.Ponds {
			if p.Sub(o.Pos).Len() < o.R+r+60 {
				return false
			}
		}
		return true
	}
	for tries := 0; len(w.Ponds) < 7 && tries < 500; tries++ {
		r := 60 + rand.Float64()*120
		p := Vec2{200 + rand.Float64()*(WorldW-400), 200 + rand.Float64()*(WorldH-400)}
		if farFromAll(p, r) {
			w.Ponds = append(w.Ponds, Pond{p, r})
		}
	}
	for tries := 0; len(w.Bushes) < Cfg.FoodBushes && tries < 2000; tries++ {
		p := Vec2{120 + rand.Float64()*(WorldW-240), 120 + rand.Float64()*(WorldH-240)}
		if !farFromAll(p, 40) {
			continue
		}
		ok := true
		for _, b := range w.Bushes {
			if p.Sub(b.Pos).Len() < 260 {
				ok = false
				break
			}
		}
		if ok {
			w.Bushes = append(w.Bushes, Bush{Pos: p, Kind: []uint8{0, 1, 3}[rand.Intn(3)], R: 22 + rand.Float64()*14, Seed: rand.Float64() * 100})
		}
	}
}

// randomCaveSite picks a random spot for the cave, away from every living
// nest, pond and bush. The required distance to nests is relaxed step by
// step so a crowded map always yields a site.
func (w *World) randomCaveSite() Vec2 {
	for _, minNest := range []float64{800, 650, 500, 0} {
		for range 400 {
			p := Vec2{300 + rand.Float64()*(WorldW-600), 300 + rand.Float64()*(WorldH-600)}
			ok := p.Sub(w.Cave).Len() > 500
			ok = ok && p.Sub(w.Shop).Len() > 500
			for _, o := range w.Ponds {
				ok = ok && p.Sub(o.Pos).Len() > o.R+330
			}
			for _, b := range w.Bushes {
				ok = ok && p.Sub(b.Pos).Len() > b.R+300
			}
			for _, c := range w.Colonies {
				ok = ok && (!c.Alive || p.Sub(c.Pos).Len() > minNest)
			}
			if ok {
				return p
			}
		}
	}
	return w.Cave
}

// moveCave collapses the cave and opens it somewhere else, so no nest stays
// the closest one to the monsters for the whole game.
func (w *World) moveCave() {
	old := w.Cave
	w.Cave = w.randomCaveSite()
	if w.Cave == old {
		return
	}
	w.emit(Event{Kind: EvCaveMove, Pos: w.Cave, Pts: []Vec2{old}})
	w.log("The cave collapses, a new rift opens elsewhere.", [3]float64{1, 0.55, 0.35})
}

// freeNestSite picks a nest position far from other nests, ponds and the cave.
func (w *World) freeNestSite() Vec2 {
	best, bestScore := Vec2{WorldW / 2, WorldH / 2}, -1.0
	for range 300 {
		p := Vec2{250 + rand.Float64()*(WorldW-500), 250 + rand.Float64()*(WorldH-500)}
		if w.InPond(p, 90) {
			continue
		}
		// The cave moves every wave, so nests only keep a safety margin from
		// it instead of fleeing to the far side of the map.
		score := rand.Float64() * 200
		minD := 1e9
		for _, c := range w.Colonies {
			if c.Alive {
				minD = math.Min(minD, p.Sub(c.Pos).Len())
			}
		}
		score += math.Min(minD, 1400)
		if p.Sub(w.Cave).Len() < 700 {
			score -= 2000
		}
		if w.Shop != (Vec2{}) && p.Sub(w.Shop).Len() < 350 {
			score -= 2000
		}
		if score > bestScore {
			best, bestScore = p, score
		}
	}
	return best
}

var colonyNames = []struct {
	Name  string
	Color [3]float64
}{
	{"Ember", [3]float64{1.0, 0.42, 0.18}},
	{"Azure", [3]float64{0.28, 0.6, 1.0}},
	{"Sylvan", [3]float64{0.42, 0.95, 0.38}},
	{"Storm", [3]float64{0.78, 0.48, 1.0}},
	{"Amber", [3]float64{1.0, 0.8, 0.22}},
	{"Opal", [3]float64{0.35, 0.98, 0.9}},
	{"Bramble", [3]float64{1.0, 0.34, 0.62}},
	{"Frostmoon", [3]float64{0.75, 0.88, 1.0}},
	{"Ash", [3]float64{0.9, 0.62, 0.5}},
	{"Myrtle", [3]float64{0.6, 1.0, 0.7}},
}

func (w *World) foundColony(id int, pos Vec2, pool []*neat.Genome) *Colony {
	var name string
	var col [3]float64
	for _, n := range rand.Perm(len(colonyNames)) {
		if !w.usedNames[colonyNames[n].Name] {
			name, col = colonyNames[n].Name, colonyNames[n].Color
			break
		}
	}
	if name == "" {
		name, col = "Wanderer", [3]float64{0.8, 0.8, 0.8}
	}
	w.usedNames[name] = true
	for k := range MaxColonies {
		w.lastFight[id][k], w.lastFight[k][id] = w.Tick, w.Tick
	}
	c := &Colony{
		ID: id, Name: name, Color: col, Pos: pos, Spouse: -1, LastQueenEgg: w.Tick,
		Food: Cfg.StartFood, HP: Cfg.NestMaxHP, Alive: true,
		BiasElem: rand.Intn(NumElements), Founded: w.Tick,
	}
	for _, g := range pool {
		cp := g.Copy()
		if len(cp.Traits) != NumTraits {
			cp.Traits = neat.RandomTraits(NumTraits)
		}
		c.Hall = append(c.Hall, cp)
	}
	if id < len(w.Colonies) {
		w.Colonies[id] = c
	} else {
		w.Colonies = append(w.Colonies, c)
	}
	for c.Born < Cfg.StartAnts {
		a := w.hatch(c)
		a.Age = rand.Intn(a.MaxAge / 3)
		a.foundAt = a.Age
		a.Pos = pos.Add(Polar(rand.Float64()*math.Pi*2, rand.Float64()*NestRadius*2))
		a.Prev = a.Pos
	}
	w.emit(Event{Kind: EvColonyFound, Pos: pos, Colony: id})
	return c
}

func (w *World) newGenome(c *Colony) *neat.Genome {
	pool := append(append([]*neat.Genome{}, c.Hall...), c.Recent...)
	var g *neat.Genome
	if s := w.SpouseOf(c.ID); s >= 0 && len(pool) > 0 && len(w.Colonies[s].Hall) > 0 && rand.Float64() < 0.25 {
		// Married colonies raise some children of both lineages.
		ga, gb := weightedPick(pool), weightedPick(w.Colonies[s].Hall)
		if gb.Fitness > ga.Fitness {
			ga, gb = gb, ga
		}
		g = neat.Crossover(ga, gb, 0)
		g.Mutate()
	} else if len(pool) >= 2 && Advanced() {
		g = w.breedAdvanced(c, pool)
	} else if len(pool) >= 2 {
		g = breedFromPool(pool)
	} else {
		g = neat.NewGenomeWithHidden(0, AntInputs, AntOutputs, Cfg.AntHidden)
		g.Traits = neat.RandomTraits(NumTraits)
		for e := range NumElements {
			g.Traits[e] *= 0.55
		}
		g.Traits[c.BiasElem] = 0.6 + rand.Float64()*0.4
		g.Traits[TraitInstinct] = Cfg.InstinctMin + rand.Float64()*(Cfg.InstinctMax-Cfg.InstinctMin)
		g.Traits[TraitDiligence] = 0.4 + rand.Float64()*0.6
		if rand.Float64() < 0.35 {
			g.Traits[rand.Intn(NumElements)] = 0.7 + rand.Float64()*0.3
		}
		g.Mutate()
	}
	if len(g.Traits) != NumTraits {
		g.Traits = neat.RandomTraits(NumTraits)
	}
	w.nextID++
	g.ID = w.nextID
	g.Fitness = 0
	return g
}

func (w *World) hatch(c *Colony) *Ant {
	g := w.newGenome(c)
	// instinct_max caps every lineage, not only fresh genomes: evolution tends
	// to push instinct up, and the cap is written back so it is inherited.
	g.Traits[TraitInstinct] = math.Min(g.Traits[TraitInstinct], Cfg.InstinctMax)
	t := g.Traits
	elem, best := 0, -1.0
	for e := range NumElements {
		if t[e] > best {
			elem, best = e, t[e]
		}
	}
	size := 0.7 + t[TraitSize]*0.9
	w.nextID++
	a := &Ant{
		ID:        w.nextID,
		Pos:       c.Pos.Add(Polar(rand.Float64()*math.Pi*2, rand.Float64()*10)),
		Angle:     rand.Float64() * math.Pi * 2,
		Colony:    c.ID,
		Genome:    g,
		Net:       g.BuildNetwork(),
		Energy:    0.9,
		Mana:      rand.Float64() * 0.5,
		Element:   elem,
		Power:     0.45 + best*1.05,
		Size:      size,
		SpeedMul:  (1.25 - 0.3*size) * (0.8 + t[TraitSpeed]*0.45),
		Instinct:  t[TraitInstinct],
		Courage:   t[TraitCourage],
		Curiosity: t[TraitCuriosity],
		Loyalty:   t[TraitLoyalty],
		Drive:     t[TraitDiligence],
		MaxAge:    Cfg.AntMaxAge/2 + rand.Intn(Cfg.AntMaxAge/2),
		Alive:     true,
		wander:    rand.Float64() * 100,
	}
	a.BornWealth, a.BornPop, a.BornFounded = c.Wealth, c.Pop, c.Founded
	a.MaxHP = 10 * size * size
	a.HP = a.MaxHP
	a.Prev = a.Pos
	a.Heading = a.Angle
	c.Born++
	w.Ants = append(w.Ants, a)
	w.maybeFoundKiosk(a, c)
	w.emit(Event{Kind: EvHatch, Pos: a.Pos, Colony: c.ID})
	return a
}

func (w *World) emit(e Event) {
	if len(w.Events) < 3000 {
		w.Events = append(w.Events, e)
	}
}

func (w *World) log(s string, col [3]float64) {
	w.Log = append(w.Log, LogMsg{s, w.Tick, col})
	if len(w.Log) > 40 {
		w.Log = w.Log[len(w.Log)-40:]
	}
}

func (w *World) dropFood(b Bush) {
	if len(w.Foods) >= Cfg.FoodMaxItems {
		return
	}
	n := 3 + rand.Intn(5)
	for range n {
		p := b.Pos.Add(Polar(rand.Float64()*math.Pi*2, b.R+rand.Float64()*55))
		if w.InPond(p, 4) {
			continue
		}
		w.placeFood(clampWorld(p), b.Kind)
	}
}

func (w *World) scatterFood(p Vec2, n int, kind uint8) {
	for range n {
		if len(w.Foods) >= Cfg.FoodMaxItems+200 {
			return
		}
		q := p.Add(Polar(rand.Float64()*math.Pi*2, rand.Float64()*28))
		w.placeFood(clampWorld(q), kind)
	}
}

// placeFood lays a food item on the ground, unless it falls within the food
// gap of a living nest: it is then lost, so food never appears where a colony
// could get it without walking.
func (w *World) placeFood(p Vec2, kind uint8) {
	gap := w.CurrentFoodGap()
	for _, c := range w.Colonies {
		if c.Alive && c.Pos.Sub(p).Len() < gap {
			return
		}
	}
	w.Foods = append(w.Foods, Food{Pos: p, Kind: kind, Rot: rand.Float64() * 6.28})
}

func (w *World) InPond(p Vec2, margin float64) bool {
	for _, o := range w.Ponds {
		r := o.R + margin
		if r > 0 && p.Sub(o.Pos).Len2() < r*r {
			return true
		}
	}
	return false
}

func (w *World) pushOutOfPonds(p Vec2) Vec2 {
	for _, o := range w.Ponds {
		d := p.Sub(o.Pos)
		if l2 := d.Len2(); l2 < o.R*o.R && l2 > 0 {
			p = o.Pos.Add(d.Scale(o.R / math.Sqrt(l2)))
		}
	}
	return p
}

// --- Spatial structures ---

func cellOf(p Vec2) int {
	x := int(p.X) / Cell
	y := int(p.Y) / Cell
	if x < 0 || y < 0 || x >= GW || y >= GH {
		return -1
	}
	return y*GW + x
}

func hashOf(p Vec2) (int, int) {
	return clampI(int(p.X)/HashCell, 0, HW-1), clampI(int(p.Y)/HashCell, 0, HH-1)
}

func (w *World) rebuildGrids() {
	w.countGen++ // invalidates every cell's counts at once, no clearing
	w.foodIdx.build(len(w.Foods), func(i int) int32 {
		f := &w.Foods[i]
		if f.Taken {
			return -1
		}
		c := cellOf(f.Pos)
		if c >= 0 {
			if cc := w.touch(c); cc.food < 255 {
				cc.food++
			}
		}
		return int32(c)
	})
	w.antHash.build(len(w.Ants), func(i int) int32 {
		a := w.Ants[i]
		if !a.Alive {
			return -1
		}
		if c := cellOf(a.Pos); c >= 0 {
			if cc := w.touch(c); cc.ants[a.Colony] < 255 {
				cc.ants[a.Colony]++
				cc.total++
			}
		}
		hx, hy := hashOf(a.Pos)
		return int32(hy*HW + hx)
	})
	w.monHash.build(len(w.Monsters), func(i int) int32 {
		m := w.Monsters[i]
		if !m.Alive {
			return -1
		}
		if c := cellOf(m.Pos); c >= 0 {
			if cc := w.touch(c); cc.mon < 255 {
				cc.mon++
			}
		}
		hx, hy := hashOf(m.Pos)
		return int32(hy*HW + hx)
	})
}

// forAntsNear calls fn for every living ant within r of p.
func (w *World) forAntsNear(p Vec2, r float64, fn func(a *Ant, d float64)) {
	x0, y0 := hashOf(Vec2{p.X - r, p.Y - r})
	x1, y1 := hashOf(Vec2{p.X + r, p.Y + r})
	r2 := r * r
	for y := y0; y <= y1; y++ {
		for x := x0; x <= x1; x++ {
			for i := w.antHash.first(y*HW + x); i >= 0; i = w.antHash.next[i] {
				a := w.Ants[i]
				if !a.Alive {
					continue
				}
				dx, dy := a.Pos.X-p.X, a.Pos.Y-p.Y
				if d2 := dx*dx + dy*dy; d2 <= r2 {
					fn(a, math.Sqrt(d2))
				}
			}
		}
	}
}

func (w *World) forMonstersNear(p Vec2, r float64, fn func(m *Monster, d float64)) {
	x0, y0 := hashOf(Vec2{p.X - r - 30, p.Y - r - 30})
	x1, y1 := hashOf(Vec2{p.X + r + 30, p.Y + r + 30})
	for y := y0; y <= y1; y++ {
		for x := x0; x <= x1; x++ {
			for i := w.monHash.first(y*HW + x); i >= 0; i = w.monHash.next[i] {
				m := w.Monsters[i]
				if !m.Alive {
					continue
				}
				dx, dy := m.Pos.X-p.X, m.Pos.Y-p.Y
				reach := r + m.Radius
				if d2 := dx*dx + dy*dy; d2 <= reach*reach {
					fn(m, maxf(math.Sqrt(d2)-m.Radius, 0))
				}
			}
		}
	}
}

// minBrainChunk is the smallest slice of brains worth handing to a worker.
var minBrainChunk = 8

// clockFreq sets the period of World.Clock: about 209 ticks.
const clockFreq = 0.03

// findHorizon is how many ticks without finding food saturate that input.
const findHorizon = 3000.0

// pain is the share of max HP lost since the last thought, a quarter of it
// already saturating. Healing gives no negative pain.
func pain(felt, hp, maxHP float64) float64 {
	return ClampF(4*(felt-hp)/maxHP, 0, 1)
}

var senseDirs = [NumSenseDirs]float64{0, math.Pi / 3, 2 * math.Pi / 3, math.Pi, -2 * math.Pi / 3, -math.Pi / 3}

var senseCos, senseSin = func() (c, s [NumSenseDirs]float64) {
	for i, a := range senseDirs {
		c[i], s[i] = math.Cos(a), math.Sin(a)
	}
	return
}()

func (w *World) fillInputs(a *Ant, in []float64) {
	clear(in)
	vis := Cfg.AntVision * (0.85 + 0.3*a.Size)
	c0, s0 := math.Cos(a.Angle), math.Sin(a.Angle)
	trail, alarm := w.Trail[a.Colony], w.Alarm[a.Colony]
	spouse := w.SpouseOf(a.Colony)
	for d := range NumSenseDirs {
		// Rotate the precomputed sensor direction by the ant's heading.
		ca := c0*senseCos[d] - s0*senseSin[d]
		sa := s0*senseCos[d] + c0*senseSin[d]
		base := d * SenseCh
		for step := 1; step <= 3; step++ {
			r := float64(step) * vis / 3
			c := cellOf(Vec2{a.Pos.X + ca*r, a.Pos.Y + sa*r})
			if c < 0 {
				continue
			}
			inv := 1 - float64(step-1)/3
			in[base+1] = maxf(in[base+1], float64(trail[c]))
			in[base+2] = maxf(in[base+2], float64(alarm[c]))
			if spouse >= 0 {
				in[base+1] = maxf(in[base+1], 0.8*float64(w.Trail[spouse][c]))
				in[base+2] = maxf(in[base+2], float64(w.Alarm[spouse][c]))
			}
			cc := &w.counts[c]
			if cc.stamp != w.countGen {
				continue // nothing in this cell this tick
			}
			if cc.food > 0 {
				in[base] = maxf(in[base], inv)
			}
			ally := int(cc.ants[a.Colony])
			if spouse >= 0 {
				ally += int(cc.ants[spouse])
			}
			if enemy := int(cc.total) - ally; enemy > 0 {
				in[base+3] = maxf(in[base+3], inv*minf(1, float64(enemy)/3))
			}
			if cc.mon > 0 {
				in[base+4] = maxf(in[base+4], inv)
			}
			if ally > 0 {
				in[base+5] = maxf(in[base+5], inv*minf(1, float64(ally)/4))
			}
		}
	}
	b := NumSenseDirs * SenseCh
	col := w.Colonies[a.Colony]
	to := col.Pos.Sub(a.Pos)
	if d := to.Len(); d > 1e-6 {
		// Bearing of the nest in the ant's frame, by rotation (no atan2).
		in[b] = (to.X*c0 + to.Y*s0) / d
		in[b+1] = (to.Y*c0 - to.X*s0) / d
		in[b+2] = minf(d/1500, 1)
	}
	in[b+3] = BoolF(a.Carrying)
	in[b+4] = a.Energy
	in[b+5] = a.HP / a.MaxHP
	in[b+6] = a.Mana
	in[b+7] = w.Clock
	in[b+8] = BoolF(w.WaveAlive > 0)
	buffs, debuffs := a.Fx.Count()
	in[b+9] = minf(1, float64(buffs)/2)
	in[b+10] = minf(1, float64(debuffs)/2)
	in[b+11] = float64(a.Age) / float64(max(1, a.MaxAge))
	in[b+12] = pain(a.feltHP, a.HP, a.MaxHP)
	a.feltHP = a.HP
	in[b+13] = minf(1, float64(a.Age-a.foundAt)/findHorizon)
	in[b+14] = col.Food / (col.Food + Cfg.StartFood)
	in[b+15] = col.HP / Cfg.NestMaxHP
	t := in[b+AntStateIn:]
	w.fillTargets(a, t, c0, s0)
	w.fillBush(a, t[TargetIn+3:], c0, s0)
	// Goal vector: the nest while carrying, else the nearest food, else the
	// nearest bush with food. A single link from its sine to the turn output
	// already forages; the brain is free to learn when to follow it.
	g := t[TargetIn:]
	switch bush := t[TargetIn+3:]; {
	case a.Carrying:
		g[0], g[1], g[2] = in[b], in[b+1], 1-in[b+2]
	case t[2] > 0:
		g[0], g[1], g[2] = t[0], t[1], t[2]
	default:
		g[0], g[1], g[2] = bush[0], bush[1], bush[2]*0.5
	}
}

// updateBushFood counts the food lying around each bush, every 16 ticks.
func (w *World) updateBushFood() {
	if w.Tick%16 != 0 && len(w.bushFood) == len(w.Bushes) {
		return
	}
	w.bushFood = append(w.bushFood[:0], make([]int, len(w.Bushes))...)
	for i := range w.Foods {
		f := &w.Foods[i]
		if f.Taken {
			continue
		}
		for j := range w.Bushes {
			if bu := &w.Bushes[j]; f.Pos.Sub(bu.Pos).Len() < bu.R+60 {
				w.bushFood[j]++
				break
			}
		}
	}
}

// bushSmell is how far an ant smells the nearest bush, where food grows.
const bushSmell = 700.0

// fillBush writes the bearing (cos, sin) and proximity of the nearest bush
// with food lying around.
func (w *World) fillBush(a *Ant, in []float64, c0, s0 float64) {
	best, bx, by := bushSmell*bushSmell, 0.0, 0.0
	for i := range w.Bushes {
		if i < len(w.bushFood) && w.bushFood[i] == 0 {
			continue
		}
		dx, dy := w.Bushes[i].Pos.X-a.Pos.X, w.Bushes[i].Pos.Y-a.Pos.Y
		if d2 := dx*dx + dy*dy; d2 < best {
			best, bx, by = d2, dx, dy
		}
	}
	if d := math.Sqrt(best); d < bushSmell && d > 1e-6 {
		in[0] = (bx*c0 + by*s0) / d
		in[1] = (by*c0 - bx*s0) / d
		in[2] = 1 - d/bushSmell
	}
}

// fillTargets adds three egocentric vectors, far easier for a small network
// to use than the sector rays: the nearest food, the nearest threat and the
// nearest sister, each as (cos, sin) of its bearing relative to the heading
// and a proximity in [0,1]. Distances are compared squared; one square root
// per found target only.
func (w *World) fillTargets(a *Ant, in []float64, c0, s0 float64) {
	const (
		foodRange   = 90.0
		sisterRange = 60.0
	)
	threatRange := Cfg.ThreatRange
	put := func(i int, dx, dy, d2, rng float64) {
		d := math.Sqrt(d2)
		if d < 1e-6 {
			return
		}
		// Rotate the world offset into the ant's frame: bearing cos/sin.
		in[i] = (dx*c0 + dy*s0) / d
		in[i+1] = (dy*c0 - dx*s0) / d
		in[i+2] = 1 - d/rng
	}
	px, py := a.Pos.X, a.Pos.Y

	// Nearest food, scanning the food grid around the ant.
	bx, by, best := 0.0, 0.0, foodRange*foodRange
	cx, cy := int(px)/Cell, int(py)/Cell
	for y := max(cy-3, 0); y <= min(cy+3, GH-1); y++ {
		for x := max(cx-3, 0); x <= min(cx+3, GW-1); x++ {
			if cc := &w.counts[y*GW+x]; cc.stamp != w.countGen || cc.food == 0 {
				continue
			}
			for fi := w.foodIdx.first(y*GW + x); fi >= 0; fi = w.foodIdx.next[fi] {
				f := &w.Foods[fi]
				dx, dy := f.Pos.X-px, f.Pos.Y-py
				if d2 := dx*dx + dy*dy; d2 < best {
					bx, by, best = dx, dy, d2
				}
			}
		}
	}
	if best < foodRange*foodRange {
		put(0, bx, by, best, foodRange)
	}

	// Nearest threat (monster or hostile ant) and nearest sister, in one
	// pass over the neighbouring ants.
	tx, ty, tBest := 0.0, 0.0, threatRange*threatRange
	sx, sy, sBest := 0.0, 0.0, sisterRange*sisterRange
	hx0, hy0 := hashOf(Vec2{px - threatRange, py - threatRange})
	hx1, hy1 := hashOf(Vec2{px + threatRange, py + threatRange})
	for y := hy0; y <= hy1; y++ {
		for x := hx0; x <= hx1; x++ {
			for i := w.antHash.first(y*HW + x); i >= 0; i = w.antHash.next[i] {
				o := w.Ants[i]
				if o == a || !o.Alive {
					continue
				}
				dx, dy := o.Pos.X-px, o.Pos.Y-py
				d2 := dx*dx + dy*dy
				switch {
				case o.Colony == a.Colony:
					if d2 < sBest {
						sx, sy, sBest = dx, dy, d2
					}
				case d2 < tBest && w.hostile(a.Colony, o.Colony):
					tx, ty, tBest = dx, dy, d2
				}
			}
			for i := w.monHash.first(y*HW + x); i >= 0; i = w.monHash.next[i] {
				m := w.Monsters[i]
				if !m.Alive {
					continue
				}
				dx, dy := m.Pos.X-px, m.Pos.Y-py
				if d2 := dx*dx + dy*dy; d2 < tBest {
					tx, ty, tBest = dx, dy, d2
				}
			}
		}
	}
	if tBest < threatRange*threatRange {
		put(3, tx, ty, tBest, threatRange)
	}
	if sBest < sisterRange*sisterRange {
		put(6, sx, sy, sBest, sisterRange)
	}
}

func (w *World) stepAnt(a *Ant, o []float64) {
	copy(a.lastOut[:], o)
	col := w.Colonies[a.Colony]
	a.Prev = a.Pos
	a.Age++

	toNest := col.Pos.Sub(a.Pos)
	if (w.Tick+a.ID)%15 == 0 {
		w.chooseRole(a, o, toNest.Len())
	}
	target, hasTarget := w.roleSteer(a, toNest)
	a.wander += 0.02
	instinctTurn := math.Sin(a.wander*1.7)*0.06 + (rand.Float64()-0.5)*0.12
	if a.Role == RoleExplore {
		instinctTurn *= 0.3
	}
	if hasTarget {
		instinctTurn = ClampF(angDiff(target, a.Angle)*0.18, -0.25, 0.25)
	}
	nnTurn := (o[0]*2 - 1) * 0.3
	full := FullBrain()
	mix := ClampF(a.Instinct*Cfg.InstinctWeight, 0, 1)
	if a.Role == RoleShop {
		mix = math.Max(mix, 0.9) // an errand: head straight for the shop and back
	}
	if full {
		mix = 0 // the network alone steers
	}
	a.Angle = wrapAngle(a.Angle + nnTurn*(1-mix) + instinctTurn*mix)

	speed := (0.35 + 0.65*o[1]) * Cfg.AntSpeed * a.SpeedMul
	if a.Role == RoleFlee && !full {
		speed *= 1.25
	}
	speed *= a.Fx.SpeedMul()
	if a.Role == RoleWatch && toNest.Len() < 75 && !full {
		speed *= 0.05 // stands still, eyes on the queen
	}
	before := toNest.Len()
	a.Pos = a.Pos.Add(Polar(a.Angle, speed))
	if a.Pos.X < 4 || a.Pos.X > WorldW-4 || a.Pos.Y < 4 || a.Pos.Y > WorldH-4 {
		a.Angle += math.Pi * 0.6
		a.Pos = clampWorld(a.Pos)
	}
	a.Pos = w.pushOutOfPonds(a.Pos)
	a.Gait += speed * 0.55
	a.Heading += angDiff(a.Angle, a.Heading) * 0.25

	c := cellOf(a.Pos)
	if c >= 0 {
		if a.Carrying {
			w.Trail[a.Colony][c] = float32(math.Min(float64(w.Trail[a.Colony][c])+0.2+0.2*o[2], 1))
		} else if o[2] > 0.85 {
			w.Trail[a.Colony][c] = float32(math.Min(float64(w.Trail[a.Colony][c])+0.03, 1))
		}
		if a.Flash > 0.5 || o[5] > 0.7 && (full || a.Role == RoleFight || a.Role == RoleFlee) {
			w.Alarm[a.Colony][c] = float32(math.Min(float64(w.Alarm[a.Colony][c])+0.3, 1))
		}
	}

	switch {
	case full && !a.Carrying && a.BiteCD == 0 && o[4] > 0.6:
		// No raid role: biting at a rival nest is what plunders it.
		a.RaidTarget = -1
		for _, e := range w.Colonies {
			if e.Alive && w.hostile(a.Colony, e.ID) && e.Pos.Sub(a.Pos).Len() < NestRadius+12 {
				a.RaidTarget = e.ID
			}
		}
		w.raidNest(a)
	case !full && a.Role == RoleRaid && !a.Carrying && a.BiteCD == 0:
		w.raidNest(a)
	}

	// Food pickup
	if !a.Carrying && c >= 0 {
		for fi := w.foodIdx.first(c); fi >= 0; fi = w.foodIdx.next[fi] {
			f := &w.Foods[fi]
			if !f.Taken && f.Pos.Sub(a.Pos).Len() < 10 {
				f.Taken = true
				a.foundAt = a.Age
				if a.Energy < eatBelow {
					a.eat()
					break
				}
				a.Carrying = true
				a.CarryK = f.Kind
				a.CarryFar = col.Pos.Sub(a.Pos).Len() >= DeliveryMinDist
				if a.CarryFar {
					a.Pickups++
				}
				break
			}
		}
	}

	if a.Carrying && a.Energy < starvingEat {
		a.Carrying = false
		a.eat()
	}

	dNest := col.Pos.Sub(a.Pos).Len()
	if a.Carrying {
		a.HomeProgress += before - dNest
	}
	inNest := dNest < NestRadius
	if inNest {
		if a.Carrying {
			a.Carrying = false
			a.foundAt = a.Age
			w.earn(col, 2)
			if a.CarryFar {
				a.Delivered++
				col.Delivered++
				col.Pulse = 1
				w.emit(Event{Kind: EvDeliver, Pos: col.Pos, Colony: col.ID})
			}
		}
		if a.Energy < 0.55 && col.Food > 0.05 {
			a.Energy += 0.01
			col.Food -= 0.004
		}
		a.HP = math.Min(a.MaxHP, a.HP+0.02)
	}
	regen := Cfg.ManaRegen * (0.6 + a.Power*0.4) * (0.25 + 0.75*w.ElemCharge[a.Element])
	if inNest {
		regen *= 3
	}
	a.Mana = math.Min(1, a.Mana+regen)

	// Combat and magic
	if a.BiteCD > 0 {
		a.BiteCD--
	}
	if a.CastCD > 0 {
		a.CastCD--
	}
	a.Fx.Tick()
	if dNest < NestRadius*2.2 {
		a.Fx.Add(FxRegen, 20) // the nest soothes its own
	}
	if a.Age > a.MaxAge*85/100 {
		a.Fx[FxOld] = 1
	}
	if a.Fx.Has(FxRegen) {
		a.HP = math.Min(a.MaxHP, a.HP+0.025*a.Fx.HealMul())
	}
	if d := a.Fx.dotDamage(w.Tick+a.ID, 1); d > 0 {
		w.damageAnt(a, d, nil)
		if !a.Alive {
			return
		}
	}
	a.Flash *= 0.85
	// Reflex thresholds: lowered by instinct and by the role in hybrid mode,
	// fixed in full-brain mode.
	eager, biteT, castT := 0.0, 0.6-0.45*a.Instinct, 0.6-0.3*a.Instinct
	if a.Role == RoleFight || a.Role == RoleGuard || a.Role == RoleRescue {
		eager = 0.25
	}
	if w.OnSale() && a.Pos.Sub(w.Shop).Len() < 260 {
		eager += 0.35 // Black Friday: the parking lot turns into a brawl
	}
	if full {
		eager, biteT, castT = 0, 0.6, 0.6
	}
	stunned := a.Fx.Has(FxStun)
	w.shopStep(a, dNest)
	w.grabCan(a, inNest)
	if a.SprayCD > 0 {
		a.SprayCD--
	}
	if !stunned && a.Can > 0 && a.SprayCD == 0 && (!full || o[4] > 0.6) {
		w.spray(a)
	}
	if !stunned && a.BiteCD == 0 && o[4] > biteT-eager && (full || a.Role != RoleFlee) {
		w.antBite(a)
	}
	if !stunned && a.CastCD == 0 && a.Mana >= 0.55 && o[3] > castT-eager {
		w.castSpell(a)
	}

	a.Energy -= Cfg.AntEnergyDrain * (0.5 + 0.5*math.Pow(a.Size, 1.5)) * (0.6 + 0.4*o[1])
	switch {
	case a.HP <= 0:
		w.killAnt(a)
	case a.Age > a.MaxAge:
		w.emit(Event{Kind: EvOldAge, Pos: a.Pos, Colony: a.Colony})
		w.killAnt(a)
	case a.Energy <= 0:
		w.emit(Event{Kind: EvOldAge, Pos: a.Pos, Colony: a.Colony, Val: 1})
		w.killAnt(a)
	}
}

// raidNest lets a raider damage a rival nest and steal from its reserves.
func (w *World) raidNest(a *Ant) {
	if a.RaidTarget < 0 || a.RaidTarget >= len(w.Colonies) {
		return
	}
	e := w.Colonies[a.RaidTarget]
	if !e.Alive || e.ID == a.Colony || e.Pos.Sub(a.Pos).Len() > NestRadius+12 {
		return
	}
	a.BiteCD = 30
	w.hostileAct(a.Colony, e.ID, true)
	e.HP -= 0.8 * a.Size
	e.raidHitAt = w.Tick
	e.HitFlash = math.Max(e.HitFlash, 0.5)
	w.emit(Event{Kind: EvNestHit, Pos: a.Pos, Colony: e.ID})
	if e.Food >= 1 {
		w.lose(e, 1)
		a.Carrying, a.CarryK, a.CarryFar = true, 0, true
		a.Role = RoleReturn
	}
	if w.Tick-e.LastRaidLog > 1800 {
		e.LastRaidLog = w.Tick
		w.log(w.Colonies[a.Colony].Name+" raids the stores of "+e.Name+".", w.Colonies[a.Colony].Color)
	}
}

func (w *World) antBite(a *Ant) {
	reach := 7 + 4*a.Size
	var mt *Monster
	w.forMonstersNear(a.Pos, reach, func(m *Monster, d float64) {
		if mt == nil {
			mt = m
		}
	})
	if mt != nil {
		a.BiteCD = 22
		w.damageMonster(mt, 0.9*a.Size*a.Fx.DamageDealt(), a)
		w.emit(Event{Kind: EvBite, Pos: LerpV(a.Pos, mt.Pos, 0.5), Colony: a.Colony})
		return
	}
	var et *Ant
	w.forAntsNear(a.Pos, reach, func(b *Ant, d float64) {
		if et == nil && w.hostile(a.Colony, b.Colony) {
			et = b
		}
	})
	if et != nil {
		a.BiteCD = 22
		w.damageAnt(et, 1.1*a.Size*a.Fx.DamageDealt(), a)
		w.emit(Event{Kind: EvBite, Pos: LerpV(a.Pos, et.Pos, 0.5), Colony: a.Colony})
	}
}

func (w *World) damageAnt(a *Ant, dmg float64, by *Ant) {
	if !a.Alive {
		return
	}
	a.HP -= dmg * a.Fx.DamageTaken()
	if by != nil {
		by.Impact += dmg
		w.hostileAct(by.Colony, a.Colony, a.HP <= 0)
	}
	a.Flash = 1
	if a.HP <= 0 {
		if by != nil {
			by.Kills++
			w.Colonies[by.Colony].Kills++
		} else if a.monsterHitAt > 0 && w.Tick-a.monsterHitAt < monsterBlame {
			w.antsByMonsters++
			if m := a.hitBy; m != nil && m.Alive {
				m.Hits += monKillReward
			}
		}
		w.killAnt(a)
	}
}

// nestLost tells the threat level about a fallen nest, if monsters took it:
// nests plundered by rival ants say nothing about the monsters' strength.
func (w *World) nestLost(c *Colony) {
	if c.monsterHitAt > 0 && w.Tick-c.monsterHitAt < monsterBlame {
		w.nestsLost++
	}
}

// monsterBlame is how long a monster stays blamed for an ant's death after
// hitting her: its poison may finish the job.
const monsterBlame = 600

// monsterHit is a monster's blow: the threat level counts the ants it kills,
// apart from those lost to weather, frogs or brawls.
func (w *World) monsterHit(m *Monster, a *Ant, dmg float64) {
	a.monsterHitAt, a.hitBy = w.Tick, m
	m.Hits += monHitReward
	w.damageAnt(a, dmg, nil)
}

// A monster's fitness: a little for each blow on an ant, to guide fresh
// brains, and mostly the ants it kills (poison included). Battering a nest
// earns nothing: a still target struck again at every cooldown was by far
// the easiest score, so coevolution bred nest razers that starved and
// toppled the colonies while hardly fighting the ants.
const (
	monHitReward  = 0.25
	monKillReward = 4.0
)

func (w *World) killAnt(a *Ant) {
	if !a.Alive {
		return
	}
	a.Alive = false
	col := w.Colonies[a.Colony]
	col.Deaths++
	w.antDeaths++
	w.Learn.add(a, w.Tick)
	w.gapDeaths++
	if a.HP > 0 && a.Energy <= 0 {
		w.gapStarved++
	}
	if a.Pack && a.Stolen {
		w.ShopStock++ // the guard takes the stolen pack back
	}
	if a.HP <= 0 {
		w.emit(Event{Kind: EvAntDeath, Pos: a.Pos, Colony: a.Colony, Elem: a.Element})
		if c := cellOf(a.Pos); c >= 0 {
			w.Alarm[a.Colony][c] = 1
		}
	}
	if a.Carrying {
		w.placeFood(a.Pos, a.CarryK)
	}
	// Impact (damage dealt plus healing given) rewards every element alike.
	fit := w.antFitness(a)
	g := a.Genome.Copy()
	g.ID, g.Fitness, g.Evals = a.Genome.ID, fit, 1
	// A cloned champion adds this life to the mean of its hall record.
	merged := Advanced() && mergeEval(col.Hall, g)
	if !merged && fit > 0.5 {
		col.Recent = append(col.Recent, g)
		if len(col.Recent) > Cfg.SurvivorPool {
			col.Recent = col.Recent[1:]
		}
		if Advanced() {
			InsertHallSpeciated(&col.Hall, g, col.Species.Threshold())
		} else {
			InsertHallOfFame(&col.Hall, g)
		}
	}
}

func (w *World) updateColonies() {
	// One pass over the ants for every colony's census.
	var pop [MaxColonies]int
	var elems [MaxColonies][NumElements]int
	w.shopperCount = [MaxColonies]int{}
	for _, a := range w.Ants {
		if a.Alive {
			pop[a.Colony]++
			elems[a.Colony][a.Element]++
			if a.Role == RoleShop {
				w.shopperCount[a.Colony]++
			}
		}
	}
	for _, c := range w.Colonies {
		c.Pulse *= 0.95
		c.HitFlash *= 0.9
		if !c.Alive {
			c.DeadTimer--
			if c.DeadTimer <= 0 {
				w.refound(c.ID)
			}
			continue
		}
		c.Pop, c.ElemCount = pop[c.ID], elems[c.ID]
		c.HP = math.Min(Cfg.NestMaxHP, c.HP+0.02)
		w.updateLive(c)
		w.maybeOrderDelivery(c)
		w.updateBrood(c)
		switch {
		case c.HP <= 0 && c.Pop >= 2:
			w.nestLost(c)
			w.Falls.Relocated++
			w.relocateNest(c) // the nest is lost, not the colony
		case c.HP <= 0:
			w.nestLost(c)
			if c.monsterHitAt >= c.raidHitAt {
				w.Falls.Monsters++
				// A whole colony lost weighs on the threat like three nests.
				if c.monsterHitAt > 0 && w.Tick-c.monsterHitAt < monsterBlame {
					w.nestsLost += 2
				}
			} else {
				w.Falls.Raids++
			}
			w.fallColony(c)
		case c.Pop == 0 && c.Eggs == 0 && c.Food < Cfg.AntCost && w.Tick-c.LastQueenEgg > Cfg.QueenEggEvery+120:
			w.Falls.Starved++
			w.fallColony(c) // the queen herself has starved
		}
	}
}

// FallStats counts colony collapses by cause since the last report, and the
// nests lost while the colony survived (moved elsewhere).
type FallStats struct {
	Monsters, Raids, Starved, Relocated int
	Age                                 int // summed lifetime of the fallen colonies, in ticks
}

func (s FallStats) Total() int { return s.Monsters + s.Raids + s.Starved }

func (w *World) fallColony(c *Colony) {
	w.Falls.Age += w.Tick - c.Founded
	w.widow(c)
	c.Alive = false
	c.DeadTimer = Cfg.RefoundDelay
	for _, a := range w.Ants {
		if a.Alive && a.Colony == c.ID {
			a.HP = 0
			w.killAnt(a)
		}
	}
	w.scatterFood(c.Pos, int(math.Min(c.Food, 40)), 1)
	c.Food = 0
	w.emit(Event{Kind: EvColonyFall, Pos: c.Pos, Colony: c.ID})
	w.log("The colony "+c.Name+" has collapsed.", c.Color)
	delete(w.usedNames, c.Name)
}

// refound seeds a new colony from the fittest surviving lineage.
func (w *World) refound(id int) {
	var best *Colony
	for _, c := range w.Colonies {
		if c.Alive && (best == nil || c.Pop+c.Delivered/10 > best.Pop+best.Delivered/10) {
			best = c
		}
	}
	// By default the fallen colony's own lineage carries on (fresh genomes if
	// it left none); "best" clones the most prosperous colony, which quickly
	// makes the whole world one lineage and one element.
	source := w.Colonies[id]
	switch Cfg.RefoundFrom {
	case "best":
		source = best
	case "fresh":
		source = nil
	}
	if source != nil && len(source.Hall) < 2 {
		source = nil
	}
	// The hall carries over as it is, with the species registry when the
	// lineage is the colony's own: scrambling it with mutations threw away
	// the champions and the lineages at every collapse. Births bring the
	// variation.
	var pool []*neat.Genome
	if source != nil {
		pool = source.Hall
	}
	var species Speciation
	if source == w.Colonies[id] {
		species = source.Species
	}
	c := w.foundColony(id, w.freeNestSite(), pool)
	c.Species = species
	switch {
	case source == best && best != nil:
		w.log("A queen of "+best.Name+" founds the colony "+c.Name+".", c.Color)
	case source != nil:
		w.log("A surviving queen revives her lineage: the colony "+c.Name+" is reborn.", c.Color)
	default:
		w.log("A wandering queen founds the colony "+c.Name+".", c.Color)
	}
}

func (w *World) updateWeather() {
	w.RainTimer--
	if w.RainTimer <= 0 {
		if w.Raining {
			w.Raining, w.Storm = false, false
			w.RainTimer = 6000 + rand.Intn(9000)
		} else if Cfg.WeatherRain {
			w.Raining = true
			w.Storm = rand.Float64() < 0.5
			w.RainTimer = 2400 + rand.Intn(3000)
			if w.Storm {
				w.log("A storm rumbles over the meadow.", [3]float64{0.7, 0.7, 1})
			} else {
				w.log("The rain washes away the pheromone trails.", [3]float64{0.6, 0.75, 0.9})
			}
		} else {
			w.RainTimer = 6000
		}
	}
	if w.Storm && w.Tick-w.LastStrike > 400 && rand.Float64() < 0.004 {
		w.LastStrike = w.Tick
		p := Vec2{rand.Float64() * WorldW, rand.Float64() * WorldH}
		w.emit(Event{Kind: EvStrike, Pos: p})
		w.forAntsNear(p, 45, func(a *Ant, d float64) { w.damageAnt(a, 6, nil) })
		w.forMonstersNear(p, 45, func(m *Monster, d float64) { w.damageMonster(m, 30, nil) })
	}
}

func (w *World) updateFoodAndPhero() {
	if w.Tick%Cfg.FoodSpawnEvery == 0 && len(w.Bushes) > 0 {
		w.dropFood(w.Bushes[rand.Intn(len(w.Bushes))])
	}
	if w.Tick%16 == 0 {
		for i := range w.Foods {
			if f := &w.Foods[i]; !f.Taken {
				f.Age += 16
				if f.Age > 40000 {
					f.Taken = true
				}
			}
		}
	}
	// Evaporation, batched every 8 ticks (same rate as per-tick decay).
	if w.Tick%8 == 0 {
		dec := float32(math.Pow(Cfg.PheroDecay, 8))
		if w.Raining {
			dec *= 0.99 * 0.99
		}
		decA := dec * 0.97 * 0.97
		for c := range len(w.Colonies) {
			t, al := w.Trail[c], w.Alarm[c][:len(w.Trail[c])]
			for i, v := range t {
				v *= dec
				if v < 0.004 {
					v = 0
				}
				t[i] = v
				u := al[i] * decA
				if u < 0.004 {
					u = 0
				}
				al[i] = u
			}
		}
	}
}

func (w *World) cleanup() {
	if w.Tick%30 != 0 {
		return
	}
	ants := w.Ants[:0]
	for _, a := range w.Ants {
		if a.Alive {
			ants = append(ants, a)
		}
	}
	clear(w.Ants[len(ants):])
	w.Ants = ants
	mons := w.Monsters[:0]
	for _, m := range w.Monsters {
		if m.Alive {
			mons = append(mons, m)
		}
	}
	clear(w.Monsters[len(mons):])
	w.Monsters = mons
	if w.Tick%600 == 0 {
		foods := w.Foods[:0]
		for _, f := range w.Foods {
			if !f.Taken {
				foods = append(foods, f)
			}
		}
		w.Foods = foods
	}
}

// --- Helpers ---

func angDiff(a, b float64) float64 {
	d := a - b
	if d >= -math.Pi && d <= math.Pi {
		return d // the common case: no Mod needed
	}
	d = math.Mod(a-b+math.Pi, 2*math.Pi)
	if d < 0 {
		d += 2 * math.Pi
	}
	return d - math.Pi
}

// wrapAngle keeps an angle in [-π, π] so angle differences stay cheap.
func wrapAngle(a float64) float64 {
	if a > math.Pi {
		a -= 2 * math.Pi
	} else if a < -math.Pi {
		a += 2 * math.Pi
	}
	return a
}

func clampWorld(p Vec2) Vec2 {
	return Vec2{ClampF(p.X, 4, WorldW-4), ClampF(p.Y, 4, WorldH-4)}
}

func ClampF(v, lo, hi float64) float64 {
	return math.Max(lo, math.Min(hi, v))
}

func clampI(v, lo, hi int) int {
	return max(lo, min(hi, v))
}

func Mixc(a, b [3]float64, t float64) [3]float64 {
	return [3]float64{a[0] + (b[0]-a[0])*t, a[1] + (b[1]-a[1])*t, a[2] + (b[2]-a[2])*t}
}

// LearnStats sums what the ants that died did with their lives, the honest
// measure of how well brains forage (the hall's top fitness is an outlier).
// The headless report reads and resets it.
type LearnStats struct {
	Deaths, Starved, ByMonsters, Delivered, MonsterKills, Life int
}

func (s *LearnStats) add(a *Ant, tick int) {
	s.Deaths++
	if a.HP > 0 && a.Energy <= 0 {
		s.Starved++
	}
	if a.HP <= 0 && a.monsterHitAt > 0 && tick-a.monsterHitAt < monsterBlame {
		s.ByMonsters++
	}
	s.Delivered += a.Delivered
	s.MonsterKills += a.MonsterKills
	s.Life += a.Age
}

func BoolF(b bool) float64 {
	if b {
		return 1
	}
	return 0
}

func InsertHallOfFame(hall *[]*neat.Genome, g *neat.Genome) {
	pos := len(*hall)
	for i, h := range *hall {
		if g.Fitness > h.Fitness {
			pos = i
			break
		}
	}
	if pos >= Cfg.HallOfFameSize {
		return
	}
	*hall = append(*hall, nil)
	copy((*hall)[pos+1:], (*hall)[pos:])
	(*hall)[pos] = g
	if len(*hall) > Cfg.HallOfFameSize {
		*hall = (*hall)[:Cfg.HallOfFameSize]
	}
}

func weightedPick(pool []*neat.Genome) *neat.Genome {
	total := 0.0
	for _, g := range pool {
		total += g.Fitness + 1
	}
	r := rand.Float64() * total
	for _, g := range pool {
		r -= g.Fitness + 1
		if r <= 0 {
			return g
		}
	}
	return pool[len(pool)-1]
}

func breedFromPool(pool []*neat.Genome) *neat.Genome {
	a, b := weightedPick(pool), weightedPick(pool)
	var child *neat.Genome
	if a == b {
		child = a.Copy()
	} else {
		if b.Fitness > a.Fitness {
			a, b = b, a
		}
		child = neat.Crossover(a, b, 0)
	}
	child.Mutate()
	return child
}

// updateBrood: the queen turns spare food into eggs and keeps a reserve of
// them; the surplus hatches one by one, and the reserve hatches in an
// emergency (fed by their yolk, so even in a famine). A queen with no food at
// all still lays an egg now and then from her own body.
func (w *World) updateBrood(c *Colony) {
	if c.SpawnCD > 0 {
		c.SpawnCD--
		return
	}
	c.SpawnCD = 30
	room := c.Pop+c.Eggs < Cfg.MaxAntsPerNest+Cfg.EggMax
	switch {
	case c.Eggs < Cfg.EggMax && room && c.Food >= Cfg.AntCost+8+w.savings(c),
		c.Pop < 8 && c.Eggs < 3 && c.Food >= Cfg.AntCost: // emergency: spend what is left
		c.Food -= Cfg.AntCost
		c.Eggs++
	case c.Eggs < Cfg.EggMax && w.Tick-c.LastQueenEgg > Cfg.QueenEggEvery:
		c.LastQueenEgg = w.Tick
		c.Eggs++
	}
	// The queen keeps a reserve of eggs for hard times; the surplus hatches.
	emergency := c.Pop < 8 || c.Food < Cfg.AntCost
	if c.Eggs > 0 && c.Pop < Cfg.MaxAntsPerNest && (c.Eggs > Cfg.EggReserve || emergency) {
		c.Eggs--
		w.hatch(c)
		c.Generation++
	}
}

// relocateNest: when monsters or raiders destroy the nest of a colony that
// still has workers, the queen flees and digs a new nest nearby, as far as
// possible from the monsters. Eggs are lost and half the reserves with them.
func (w *World) relocateNest(c *Colony) {
	old := c.Pos
	best, bestScore := old, -1.0
	for range 60 {
		p := clampWorld(old.Add(Polar(rand.Float64()*2*math.Pi, 350+rand.Float64()*350)))
		if w.InPond(p, 90) || p.Sub(w.Cave).Len() < 600 || p.Sub(w.Shop).Len() < 300 {
			continue
		}
		score := 1e6
		for _, m := range w.Monsters {
			if m.Alive {
				score = math.Min(score, m.Pos.Sub(p).Len())
			}
		}
		for _, o := range w.Colonies {
			if o.Alive && o != c && o.Pos.Sub(p).Len() < 400 {
				score -= 1e5
			}
		}
		score = math.Min(score, 2000) + rand.Float64()*100
		if score > bestScore {
			best, bestScore = p, score
		}
	}
	c.Pos = best
	c.HP = Cfg.NestMaxHP * 0.6
	// The loss goes through the wealth ledger (eggs at their cost), so the
	// colony-level fitness feels a fallen nest and defending it pays.
	w.lose(c, c.Food*0.5)
	c.Wealth -= float64(c.Eggs) * Cfg.AntCost
	c.Eggs = 0
	c.Moves++
	w.emit(Event{Kind: EvNestMove, Pos: best, Colony: c.ID, Pts: []Vec2{old}})
	w.log(fmt.Sprintf("The nest of %s is destroyed: the colony moves.", c.Name), c.Color)
}

// V builds a Vec2 from its coordinates.
func V(x, y float64) Vec2 { return Vec2{x, y} }
