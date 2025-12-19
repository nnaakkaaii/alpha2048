package domain

import (
	"math/rand"
	"testing"
)

func TestGameMove(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	game := NewGame(rng)

	if game.Score() != 0 {
		t.Errorf("initial score should be 0, got %d", game.Score())
	}

	directions := []Direction{Left, Right, Up, Down}
	for i := 0; i < 10; i++ {
		game.Move(directions[i%4])
	}
}

func TestGameScoreIncreases(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	game := &Game{
		board: NewBoardFromCells([4][4]int{
			{2, 2, 0, 0},
			{0, 0, 0, 0},
			{0, 0, 0, 0},
			{0, 0, 0, 0},
		}),
		score: 0,
		rng:   rng,
	}

	moved := game.Move(Left)
	if !moved {
		t.Error("expected move to succeed")
	}

	if game.Score() != 4 {
		t.Errorf("expected score 4, got %d", game.Score())
	}
}
