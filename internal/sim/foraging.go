package sim

import "math"

// Foraging helpers.
//
// A hungry ant eats what she finds instead of carrying it home, and a
// starving carrier eats her load. Looking for food then keeps the ant herself
// alive, a dense and natural signal that no reward has to fake. Eating on the
// spot is much less efficient than feeding at the nest (a delivered item is
// worth about 5 energy there), so a colony still does better by bringing food
// home.
//
// The distance below which no food appears around a nest is a curriculum:
// it starts at DeliveryMinDist and grows up to FoodNestGap as the colonies
// learn to feed themselves, measured by the share of ants dying of hunger.

const (
	eatBelow    = 0.35 // energy under which an ant eats what she picks up
	starvingEat = 0.12 // energy under which a carrier eats her load
	eatGain     = 0.4

	foodGapStarve = 0.25 // share of deaths by hunger the curriculum aims at
	foodGapK      = 0.4
)

// eat feeds the ant with one food item.
func (a *Ant) eat() {
	a.Energy = math.Min(1, a.Energy+eatGain)
	a.Eaten++
}

// CurrentFoodGap is the current distance around living nests where no food appears.
func (w *World) CurrentFoodGap() float64 {
	if !Cfg.FoodGapAdaptive {
		return FoodNestGap
	}
	return w.FoodGap
}

// adaptFoodGap runs at each new wave: the gap grows while few ants starve
// and shrinks when hunger kills too many.
func (w *World) adaptFoodGap() {
	if w.gapDeaths >= 5 {
		starve := float64(w.gapStarved) / float64(w.gapDeaths)
		w.FoodGap *= math.Exp(foodGapK * (foodGapStarve - starve))
	}
	w.FoodGap = ClampF(w.FoodGap, DeliveryMinDist, FoodNestGap)
	w.gapDeaths, w.gapStarved = 0, 0
}
