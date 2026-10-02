package sim

import (
	"fmt"
	"math"
	"math/rand"
)

// The Wallmart and the insecticide economy.
//
// Colonies spend food reserves on packs of spray cans. Shoppers take the money
// at the nest, buy at the Wallmart (or at a rival colony's cooperative kiosk),
// and bring the pack home. Bold, disloyal ants shoplift instead and get chased
// by the security guard. Rich colonies can order a scooter delivery. Once a
// day the Wallmart runs a Black Friday: prices crash, stock floods in and the
// parking lot turns into a brawl.

const (
	ShopW, ShopH = 150.0, 96.0 // building and parking footprint
	shopReach    = 70.0        // how close an ant must get to shop
	kioskReach   = 40.0
	sprayRange   = 75.0
	SprayCone    = 0.65 // half-angle in radians
	saleLength   = 1800
	maxShopStock = 30
)

const (
	SrcWallmart = iota
	SrcKiosk
)

type Kiosk struct {
	Pos     Vec2
	Owner   int
	Stock   int
	Price   float64
	Sold    int
	Total   int // packs sold since opening
	Stocked int // packs delivered by the owner
	Alive   bool
	Opened  int
}

func (w *World) placeShop() {
	w.Shop = Vec2{WorldW / 2, WorldH / 2}
	w.ShopStock = 6
	w.NextSale = int(Cfg.DayLength*60*0.6) + rand.Intn(int(Cfg.DayLength*60*0.4))
	for _, minNest := range []float64{600, 450, 300, 0} {
		for range 400 {
			p := Vec2{300 + rand.Float64()*(WorldW-600), 300 + rand.Float64()*(WorldH-600)}
			ok := p.Sub(w.Cave).Len() > 600
			for _, o := range w.Ponds {
				ok = ok && p.Sub(o.Pos).Len() > o.R+160
			}
			for _, b := range w.Bushes {
				ok = ok && p.Sub(b.Pos).Len() > b.R+120
			}
			for _, c := range w.Colonies {
				ok = ok && (!c.Alive || p.Sub(c.Pos).Len() > minNest)
			}
			if ok {
				w.Shop = p
				w.Guard = &Guard{Pos: p.Add(Vec2{ShopW * 0.6, 0}), Home: p}
				return
			}
		}
	}
	w.Guard = &Guard{Pos: w.Shop, Home: w.Shop}
}

func (w *World) OnSale() bool { return Cfg.BlackFriday && w.Tick < w.SaleUntil }

// ShopPrice is the Wallmart price right now.
func (w *World) ShopPrice() float64 {
	if w.OnSale() {
		return math.Round(Cfg.InsecticidePrice * 0.3)
	}
	return Cfg.InsecticidePrice
}

// updateShop runs the store: restocking, Black Friday, the kiosk market.
func (w *World) updateShop() {
	if !Cfg.ShopEnabled {
		return
	}
	if w.Tick%1200 == 0 && w.ShopStock < 8 {
		w.ShopStock++
	}
	// Resistance fades when monsters are no longer sprayed.
	w.InsectResist = math.Max(0, w.InsectResist-0.00002)
	if Cfg.BlackFriday && w.Tick >= w.NextSale {
		w.SaleUntil = w.Tick + saleLength
		w.NextSale = w.Tick + int(Cfg.DayLength*60)
		w.ShopStock = min(maxShopStock, w.ShopStock+20)
		w.log("BLACK FRIDAY at Wallmart: insecticide at -70 %!", [3]float64{1, 0.85, 0.2})
		w.emit(Event{Kind: EvSaleStart, Pos: w.Shop})
	}
	if k := w.Kiosk; k != nil && k.Alive {
		if !w.Colonies[k.Owner].Alive {
			k.Alive = false
			w.log("The cooperative of "+w.Colonies[k.Owner].Name+" closes shop.", w.Colonies[k.Owner].Color)
		} else if w.Tick%600 == 0 {
			// Supply and demand: sales push the price up, idleness brings it down.
			if k.Sold > 0 {
				k.Price *= 1 + 0.08*float64(k.Sold)
			} else {
				k.Price *= 0.95
			}
			k.Price = ClampF(k.Price, Cfg.InsecticidePrice*1.1, Cfg.InsecticidePrice*3)
			k.Sold = 0
		}
	}
	w.updateGuard()
	w.updateCouriers()
}

// wantsInsecticide is true when a colony should buy: low on cans with monsters
// coming, restocking its own kiosk, or a bargain it cannot resist.
func (w *World) wantsInsecticide(c *Colony) bool {
	if !Cfg.ShopEnabled || c.Pop < 12 {
		return false
	}
	if w.OnSale() {
		return c.Cans < 8
	}
	if k := w.Kiosk; k != nil && k.Alive && k.Owner == c.ID && k.Stock < 4 {
		return true
	}
	danger := w.WaveAlive > 0 || w.NextWave-w.Tick < 2400
	return c.Cans < 2 && danger
}

// savings is the food a colony holds back from hatching to afford a pack
// (and, for the big ones, the delivery fee).
func (w *World) savings(c *Colony) float64 {
	if !w.wantsInsecticide(c) {
		return 0
	}
	if c.Pop >= 25 {
		return w.ShopPrice() + Cfg.DeliveryFee
	}
	return w.ShopPrice()
}

// shoppers is the colony's number of ants on an errand (census of the last tick).
func (w *World) shoppers(c *Colony) int { return w.shopperCount[c.ID] }

// shopUtility scores the shopping role.
func (w *World) shopUtility(a *Ant, threat float64) float64 {
	if !Cfg.ShopEnabled || a.Carrying || FullBrain() {
		return 0 // no scripted errands in full-brain mode (deliveries still work)
	}
	if a.Money || a.Pack || a.Thief {
		if a.Energy < 0.3 {
			return 0.2 // go home and eat first; the errand waits
		}
		return 1.6 - threat*(1-a.Courage)
	}
	col := w.Colonies[a.Colony]
	limit := 1
	if w.OnSale() {
		limit = 6
	}
	if a.Role != RoleShop && w.shoppers(col) >= limit {
		return 0
	}
	if !w.wantsInsecticide(col) || col.Food < w.ShopPrice()+3 {
		return 0
	}
	u := 0.45 + 0.6*a.Drive
	if w.OnSale() {
		u += 0.5
	}
	return u
}

// bestSource picks where to buy: the Wallmart or a rival's kiosk, weighing
// price against the walk.
func (w *World) bestSource(a *Ant) (int, float64) {
	// A long walk is a risky walk: distance weighs heavily in the choice.
	kind, price := SrcWallmart, w.ShopPrice()
	cost := price + a.Pos.Sub(w.Shop).Len()/40
	if w.ShopStock == 0 {
		cost = math.Inf(1)
	}
	if k := w.Kiosk; k != nil && k.Alive && k.Owner != a.Colony && k.Stock > 0 {
		if c := k.Price + a.Pos.Sub(k.Pos).Len()/40; c < cost {
			kind, price = SrcKiosk, k.Price
		}
	}
	return kind, price
}

func (w *World) kioskOpen() bool { return w.Kiosk != nil && w.Kiosk.Alive }

// restocksKiosk: an owner colony with enough cans sends its packs to its stall.
func (w *World) restocksKiosk(a *Ant) bool {
	return w.kioskOpen() && w.Kiosk.Owner == a.Colony && w.Colonies[a.Colony].Cans >= 2 && !a.Stolen
}

func (w *World) errandTarget(a *Ant, toNest Vec2) Vec2 {
	switch {
	case a.Money || a.Thief:
		if a.ShopAt == SrcKiosk && w.kioskOpen() {
			return w.Kiosk.Pos
		}
		return w.Shop
	case a.Pack && w.restocksKiosk(a):
		return w.Kiosk.Pos
	}
	return a.Pos.Add(toNest)
}

func (w *World) shopSteer(a *Ant, toNest Vec2) (float64, bool) {
	to := w.errandTarget(a, toNest).Sub(a.Pos)
	return math.Atan2(to.Y, to.X), true
}

// shopStep advances the errand: pay (or decide to steal), buy, bring it home.
func (w *World) shopStep(a *Ant, dNest float64) {
	if a.Role != RoleShop {
		return
	}
	col := w.Colonies[a.Colony]
	switch {
	case !a.Money && !a.Pack && !a.Thief:
		if dNest >= NestRadius {
			return
		}
		kind, price := w.bestSource(a)
		a.ShopAt, a.WaitShop = kind, 0
		// Bold, disloyal ants shoplift rather than pay: the more courage
		// outweighs loyalty in her genome, the likelier she steals.
		if kind == SrcWallmart && rand.Float64() < ClampF(1.2*(a.Courage-a.Loyalty), 0, 0.8) {
			a.Thief = true
			return
		}
		if col.Food < price {
			a.Role = RoleForage
			return
		}
		col.Food -= price
		a.Paid, a.Money = price, true

	case a.Money || a.Thief:
		kiosk := a.ShopAt == SrcKiosk && w.kioskOpen()
		if a.ShopAt == SrcKiosk && !kiosk {
			a.ShopAt = SrcWallmart // the kiosk closed on the way
		}
		at, reach := w.Shop, shopReach
		if kiosk {
			at, reach = w.Kiosk.Pos, kioskReach
		}
		if a.Pos.Sub(at).Len() > reach {
			return
		}
		a.WaitShop++
		switch {
		case kiosk && w.Kiosk.Stock > 0:
			k := w.Kiosk
			k.Stock--
			k.Sold++
			k.Total++
			w.earn(w.Colonies[k.Owner], a.Paid)
			a.Money, a.Pack = false, true
			w.emit(Event{Kind: EvShopBuy, Pos: k.Pos, Colony: a.Colony})
		case kiosk && a.WaitShop > 600:
			a.ShopAt = SrcWallmart // sold out: try the Wallmart
		case !kiosk && w.ShopStock > 0:
			w.ShopStock--
			a.Pack = true
			if a.Thief {
				a.Thief, a.Stolen = false, true
				w.emit(Event{Kind: EvStolen, Pos: a.Pos, Colony: a.Colony})
				if w.Tick-w.lastTheftLog > 1800 {
					w.lastTheftLog = w.Tick
					w.log("An ant of "+col.Name+" shoplifts at Wallmart!", col.Color)
				}
			} else {
				a.Money = false
				w.emit(Event{Kind: EvShopBuy, Pos: w.Shop, Colony: a.Colony})
			}
		}

	case a.Pack:
		if w.restocksKiosk(a) {
			if a.Pos.Sub(w.Kiosk.Pos).Len() < kioskReach {
				w.Kiosk.Stock += Cfg.InsecticidePack
				w.Kiosk.Stocked += Cfg.InsecticidePack
				a.Pack = false
				a.Role = RoleReturn
			}
			return
		}
		if dNest < NestRadius {
			a.Pack, a.Stolen = false, false
			col.Cans += Cfg.InsecticidePack
			a.Role = RoleReturn
			if w.Tick-col.LastShopLog > 2400 {
				col.LastShopLog = w.Tick
				w.log("An ant of "+col.Name+" comes back from Wallmart with insecticide.", col.Color)
			}
		}
	}
}

// maybeFoundKiosk: now and then a newborn more curious than her sisters
// opens a cooperative that resells insecticide to the other colonies.
func (w *World) maybeFoundKiosk(a *Ant, c *Colony) {
	if !Cfg.ShopEnabled || c.Pop < 20 || w.kioskOpen() || rand.Float64() > 0.015 {
		return
	}
	sum, n := 0.0, 0
	for _, b := range w.Ants {
		if b.Alive && b.Colony == c.ID {
			sum += b.Curiosity
			n++
		}
	}
	if n == 0 || a.Curiosity <= sum/float64(n) {
		return
	}
	center := Vec2{WorldW / 2, WorldH / 2}
	pos := c.Pos
	for range 50 {
		pos = LerpV(c.Pos, center, 0.3+rand.Float64()*0.2).Add(Polar(rand.Float64()*6.28, rand.Float64()*80))
		if !w.InPond(pos, 40) {
			break
		}
	}
	w.Kiosk = &Kiosk{Pos: clampWorld(pos), Owner: c.ID, Price: Cfg.InsecticidePrice * 1.5, Alive: true, Opened: w.Tick}
	w.log(fmt.Sprintf("Worker #%d of %s, more curious than her sisters, founds an insecticide cooperative!", a.ID, c.Name), c.Color)
	w.emit(Event{Kind: EvKioskOpen, Pos: w.Kiosk.Pos, Colony: c.ID})
}

// grabCan lets a defender take a spray can from the nest stock during a wave.
func (w *World) grabCan(a *Ant, inNest bool) {
	col := w.Colonies[a.Colony]
	if inNest && a.Can == 0 && col.Cans > 0 && w.WaveAlive > 0 &&
		(FullBrain() && !a.Carrying || a.Role == RoleFight || a.Role == RoleGuard || a.Role == RoleRescue) {
		col.Cans--
		a.Can = Cfg.InsecticideCharges
	}
}

// spray fires a cone of insecticide at the nearest monster. It hurts every
// creature caught in the cloud, sisters included (less so). Monsters grow
// resistant the more they are sprayed.
func (w *World) spray(a *Ant) {
	var target *Monster
	best := sprayRange
	w.forMonstersNear(a.Pos, sprayRange, func(m *Monster, d float64) {
		if d < best {
			best, target = d, m
		}
	})
	if target == nil {
		return
	}
	to := target.Pos.Sub(a.Pos)
	dir := math.Atan2(to.Y, to.X)
	a.Angle, a.Heading = dir, dir
	a.Can--
	a.SprayCD = 50
	inCone := func(p Vec2, reach float64) bool {
		d := p.Sub(a.Pos)
		l := d.Len()
		return l < reach && (l < 8 || math.Abs(angDiff(math.Atan2(d.Y, d.X), dir)) < SprayCone)
	}
	eff := 1 - w.InsectResist
	hit := 0
	w.forMonstersNear(a.Pos, sprayRange, func(m *Monster, d float64) {
		if inCone(m.Pos, sprayRange+m.Radius) {
			w.damageMonster(m, 22*a.Fx.DamageDealt()*eff, a)
			m.Fx.Add(FxPoison, int(360*eff))
			m.Fx.Add(FxWeak, int(360*eff))
			hit++
		}
	})
	w.forAntsNear(a.Pos, sprayRange, func(o *Ant, d float64) {
		if o != a && inCone(o.Pos, sprayRange) {
			dmg := 3.0
			if w.allied(a.Colony, o.Colony) {
				dmg = 1.5 // friendly fire still hurts, and a spouse remembers
			}
			w.damageAnt(o, dmg, a)
			o.Fx.Add(FxPoison, 120)
		}
	})
	w.addResistance(float64(hit) * 0.004)
	w.emit(Event{Kind: EvSpray, Pos: a.Pos, Colony: a.Colony, Val: dir})
}

// addResistance lets the monster population adapt to the insecticide.
func (w *World) addResistance(d float64) {
	before := w.InsectResist
	w.InsectResist = ClampF(w.InsectResist+d, 0, 0.85)
	for _, step := range []float64{0.25, 0.5, 0.75} {
		if before < step && w.InsectResist >= step {
			w.log(fmt.Sprintf("Monsters are now %.0f %% resistant to insecticide.", step*100), [3]float64{0.7, 0.95, 0.5})
		}
	}
}
