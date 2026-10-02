package sim

import (
	"fmt"
	"github.com/ThiraSoft/neatants/neat"
	"os"

	"gopkg.in/yaml.v3"
)

type Config struct {
	// Colonies
	Colonies        int     `yaml:"colonies"`
	StartAnts       int     `yaml:"start_ants"`
	MaxAntsPerNest  int     `yaml:"max_ants_per_nest"`
	AntCost         float64 `yaml:"ant_cost"`
	EggMax          int     `yaml:"egg_max"`
	EggReserve      int     `yaml:"egg_reserve"`
	QueenEggEvery   int     `yaml:"queen_egg_every"`
	StartFood       float64 `yaml:"start_food"`
	AntSpeed        float64 `yaml:"ant_speed"`
	AntVision       float64 `yaml:"ant_vision"`
	AntMaxAge       int     `yaml:"ant_max_age"`
	AntEnergyDrain  float64 `yaml:"ant_energy_drain"`
	AntHidden       int     `yaml:"ant_hidden"`
	ThinkEvery      int     `yaml:"think_every"`
	ThreatRange     float64 `yaml:"threat_range"`
	AimAssist       float64 `yaml:"aim_assist"`
	InstinctMin     float64 `yaml:"instinct_min"`
	InstinctMax     float64 `yaml:"instinct_max"`
	InstinctWeight  float64 `yaml:"instinct_weight"`
	BrainMode       string  `yaml:"brain_mode"`
	Selection       string  `yaml:"selection"`
	NestMaxHP       float64 `yaml:"nest_max_hp"`
	RefoundDelay    int     `yaml:"refound_delay"`
	RefoundFrom     string  `yaml:"refound_from"`
	ShapingFade     int     `yaml:"shaping_fade"`
	ShapingFloor    float64 `yaml:"shaping_floor"`
	AdaptiveWaves   bool    `yaml:"adaptive_waves"`
	WaveBalance     float64 `yaml:"wave_balance"`
	ThreatSignal    string  `yaml:"threat_signal"`
	ThreatDeaths    float64 `yaml:"threat_deaths"`
	ThreatAge       float64 `yaml:"threat_age"`
	MonsterBrains   bool    `yaml:"monster_brains"`
	FoodGapAdaptive bool    `yaml:"food_gap_adaptive"`
	HallOfFameSize  int     `yaml:"hall_of_fame_size"`
	SurvivorPool    int     `yaml:"survivor_pool_size"`
	ManaRegen       float64 `yaml:"mana_regen"`

	// Food
	FoodBushes     int `yaml:"food_bushes"`
	FoodSpawnEvery int `yaml:"food_spawn_every"`
	FoodMaxItems   int `yaml:"food_max_items"`

	// Pheromones
	PheroDecay float64 `yaml:"phero_decay"`

	// Monsters
	FirstWave        int     `yaml:"first_wave"`
	WaveInterval     int     `yaml:"wave_interval"`
	WaveBase         int     `yaml:"wave_base"`
	WaveMin          int     `yaml:"wave_min"`
	WaveGrowth       float64 `yaml:"wave_growth"`
	WaveHPGrowth     float64 `yaml:"wave_hp_growth"`
	WavePerAnt       float64 `yaml:"wave_per_ant"`
	MonsterHPMult    float64 `yaml:"monster_hp_mult"`
	BossEvery        int     `yaml:"boss_every"`
	MonsterMaxAge    int     `yaml:"monster_max_age"`
	MonsterDisease   bool    `yaml:"monster_disease"`
	DiseaseOnset     float64 `yaml:"disease_onset"`
	DiseaseContagion float64 `yaml:"disease_contagion"`
	DiseaseDamage    float64 `yaml:"disease_damage"`

	// Shop
	ShopEnabled        bool    `yaml:"shop_enabled"`
	InsecticidePrice   float64 `yaml:"insecticide_price"`
	InsecticidePack    int     `yaml:"insecticide_pack"`
	InsecticideCharges int     `yaml:"insecticide_charges"`
	DeliveryFee        float64 `yaml:"delivery_fee"`
	BlackFriday        bool    `yaml:"black_friday"`
	WeirdWeather       bool    `yaml:"weird_weather"`
	QueenLive          bool    `yaml:"queen_live"`
	Marriage           bool    `yaml:"marriage"`
	MarriageDelay      int     `yaml:"marriage_delay"`
	DivorceGrief       int     `yaml:"divorce_grief"`
	CaveMoves          bool    `yaml:"cave_moves_each_wave"`

	// Misc
	SaveEvery   int     `yaml:"save_every"`
	DayLength   float64 `yaml:"day_length_seconds"`
	WeatherRain bool    `yaml:"weather_rain"`
}

var Cfg Config

func LoadConfig(path string) {
	Cfg = Config{
		Colonies:           4,
		StartAnts:          28,
		MaxAntsPerNest:     100,
		AntCost:            5,
		EggMax:             20,
		EggReserve:         6,
		QueenEggEvery:      1200,
		StartFood:          40,
		AntSpeed:           1.6,
		AntVision:          110,
		AntMaxAge:          36000,
		AntEnergyDrain:     0.00012,
		AntHidden:          0,
		ThinkEvery:         2,
		ThreatRange:        130,
		AimAssist:          20,
		InstinctMin:        0.25,
		InstinctMax:        0.55,
		InstinctWeight:     0.85,
		BrainMode:          "hybrid",
		Selection:          "colony",
		NestMaxHP:          400,
		RefoundDelay:       1800,
		RefoundFrom:        "own",
		ShapingFade:        1000000,
		ShapingFloor:       0.3,
		AdaptiveWaves:      true,
		WaveBalance:        0.5,
		ThreatSignal:       "deaths",
		ThreatDeaths:       0.25,
		ThreatAge:          0.25,
		MonsterBrains:      true,
		FoodGapAdaptive:    true,
		HallOfFameSize:     20,
		SurvivorPool:       40,
		ManaRegen:          1.0 / 600,
		FoodBushes:         14,
		FoodSpawnEvery:     160,
		FoodMaxItems:       650,
		PheroDecay:         0.996,
		FirstWave:          2400,
		WaveInterval:       4200,
		WaveBase:           12,
		WaveMin:            6,
		WaveGrowth:         2.5,
		WaveHPGrowth:       0,
		WavePerAnt:         0.3,
		MonsterHPMult:      1,
		BossEvery:          5,
		MonsterMaxAge:      12000,
		MonsterDisease:     true,
		DiseaseOnset:       0.3,
		DiseaseContagion:   0.25,
		DiseaseDamage:      0.03,
		ShopEnabled:        true,
		InsecticidePrice:   20,
		InsecticidePack:    3,
		InsecticideCharges: 3,
		DeliveryFee:        12,
		BlackFriday:        true,
		WeirdWeather:       true,
		QueenLive:          true,
		Marriage:           true,
		MarriageDelay:      8000,
		DivorceGrief:       3,
		CaveMoves:          true,
		SaveEvery:          6000,
		DayLength:          300,
		WeatherRain:        true,
	}
	data, err := os.ReadFile(path)
	if err != nil {
		fmt.Printf("[CONFIG] %s not found, using defaults\n", path)
		return
	}
	if err := yaml.Unmarshal(data, &Cfg); err != nil {
		fmt.Printf("[CONFIG] read error %s: %v\n", path, err)
		return
	}
	fmt.Printf("[CONFIG] loaded from %s\n", path)
	neat.Advanced = Cfg.Selection != "classic"
}

// FullBrain reports whether the neural networks alone drive the ants.
func FullBrain() bool { return Cfg.BrainMode == "full" }
