package sim

import (
	"fmt"
	"math/rand"
	"os"
	"testing"
)

// TestWTF runs long games and checks every silly mechanic happens.
// Slow: runs only with NEATANTS_LONG_TESTS=1.
func TestWTF(t *testing.T) {
	if os.Getenv("NEATANTS_LONG_TESTS") == "" {
		t.Skip("NEATANTS_LONG_TESTS=1 to run this long test")
	}
	LoadConfig("config.yml")
	count := map[EventKind]int{}
	frogs, confetti, watchers, maxKioskPrice, maxResist := 0, 0, 0, 0.0, 0.0
	kioskSold, kioskStocked := 0, 0
	var w *World
	for _, seed := range []int64{9, 21} {
		rand.Seed(seed)
		w = NewWorld()
		for tick := range 80000 {
			w.Update()
			for _, e := range w.Events {
				count[e.Kind]++
				if e.Kind == EvWeirdStart {
					if int(e.Val) == WeirdFrogs {
						frogs++
					} else {
						confetti++
					}
				}
			}
			w.Events = w.Events[:0]
			if tick%20 == 0 {
				for _, a := range w.Ants {
					if a.Alive && a.Role == RoleWatch {
						watchers++
					}
				}
			}
			if w.Kiosk != nil {
				maxKioskPrice = max(maxKioskPrice, w.Kiosk.Price)
				if tick == 79999 || !w.Kiosk.Alive {
					kioskSold, kioskStocked = max(kioskSold, w.Kiosk.Total), max(kioskStocked, w.Kiosk.Stocked)
				}
			}
			maxResist = max(maxResist, w.InsectResist)
		}
	}
	fmt.Printf("Black Friday: %d · purchases: %d · thefts: %d · guard bites: %d (thieves caught %d)\n",
		count[EvSaleStart], count[EvShopBuy], count[EvStolen], count[EvGuardBite], w.Guard.Catches)
	fmt.Printf("cooperative: %d openings, max price %.1f · deliveries: %d · ants crushed (shocks): %d\n",
		count[EvKioskOpen], maxKioskPrice, count[EvCourierArrive], count[EvCourierHit])
	fmt.Printf("cooperative: %d packs received, %d sold\n", kioskStocked, kioskSold)
	fmt.Printf("frog rains: %d · confetti storms: %d · frog meals: %d\n", frogs, confetti, count[EvFrogEat])
	fmt.Printf("lives: %d · viewer samples: %d · max resistance %.0f %% · sprays %d\n",
		count[EvLiveStart], watchers, maxResist*100, count[EvSpray])
	for _, c := range w.Colonies {
		fmt.Printf("  %-10s alive=%v pop=%d stock=%.0f bombs=%d likes=%d\n", c.Name, c.Alive, c.Pop, c.Food, c.Cans, c.Likes)
	}
	// Shoplifting depends on the lineages (bold, disloyal ants), so it is not required.
	for _, k := range []EventKind{EvSaleStart, EvWeirdStart, EvFrogEat, EvLiveStart, EvKioskOpen, EvCourierArrive} {
		if count[k] == 0 {
			t.Errorf("event %d never happened", k)
		}
	}
}
