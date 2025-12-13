package main

import (
	"flag"
	"math/rand"
	"os"
	"time"

	"github.com/nnaakkaaii/alpha2048/internal/usecase"
)

func main() {
	depth := flag.Int("depth", 3, "search depth")
	delay := flag.Int("delay", 100, "delay between moves (ms)")
	useAStar := flag.Bool("astar", false, "use A* algorithm")
	useBitBoard := flag.Bool("bitboard", false, "use bitboard representation")
	quiet := flag.Bool("quiet", false, "suppress output")
	flag.Parse()

	rng := rand.New(rand.NewSource(time.Now().UnixNano()))

	config := usecase.DefaultAutoPlayConfig()
	config.MaxDepth = *depth
	config.Delay = time.Duration(*delay) * time.Millisecond
	config.UseAStar = *useAStar
	config.UseBitBoard = *useBitBoard
	config.Verbose = !*quiet

	usecase.AutoPlay(os.Stdout, rng, config)
}
