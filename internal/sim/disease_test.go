package sim

import (
	"fmt"
	"math/rand"
	"os"
	"testing"
)

// TestDiseaseEffect compares monster populations with and without cave fever.
// It is slow (late-game hordes), so it only runs with NEATANTS_LONG_TESTS=1.
func TestDiseaseEffect(t *testing.T) {
	if os.Getenv("NEATANTS_LONG_TESTS") == "" {
		t.Skip("NEATANTS_LONG_TESTS=1 to run this long test")
	}
	for _, on := range []bool{false, true} {
		LoadConfig("config.yml")
		Cfg.MonsterDisease = on
		totalAlive, samples, peak, sickPeak := 0, 0, 0, 0
		ages, deaths := 0, 0
		byDisease, byAge := 0, 0
		for seed := range 3 {
			rand.Seed(int64(seed + 1))
			w := NewWorld()
			w.Wave, w.NextWave = 60, 100 // late game: huge waves
			lastAge := map[int]int{}
			wasSick := map[int]bool{}
			for tick := range 40000 {
				w.Update()
				w.Events = w.Events[:0]
				alive, sick := 0, 0
				for _, m := range w.Monsters {
					if m.Alive {
						alive++
						lastAge[m.ID] = m.Age
						if m.Fx.Has(FxPlague) {
							sick++
							wasSick[m.ID] = true
						}
					} else if a, ok := lastAge[m.ID]; ok {
						ages += a
						deaths++
						if m.Age > m.MaxAge {
							byAge++
						} else if wasSick[m.ID] {
							byDisease++
						}
						delete(lastAge, m.ID)
					}
				}
				if tick > 5000 && tick%50 == 0 {
					totalAlive += alive
					samples++
				}
				peak = max(peak, alive)
				sickPeak = max(sickPeak, sick)
			}
		}
		label := "without disease"
		if on {
			label = "with disease"
		}
		fmt.Printf("%s: %.1f living monsters on average, peak %d, sick peak %d, mean age at death %d ticks, deaths of old age %d, deaths while sick %d\n",
			label, float64(totalAlive)/float64(samples), peak, sickPeak, ages/max(1, deaths), byAge, byDisease)
	}
}
