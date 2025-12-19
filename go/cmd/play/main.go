package main

import (
	"math/rand"
	"os"
	"time"

	"github.com/nnaakkaaii/alpha2048/go/internal/usecase"
)

func main() {
	rng := rand.New(rand.NewSource(time.Now().UnixNano()))
	usecase.PlayGame(os.Stdin, os.Stdout, rng)
}
