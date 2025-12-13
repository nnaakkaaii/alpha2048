package usecase

import (
	"fmt"
	"io"
	"math/rand"
	"time"

	"github.com/nnaakkaaii/alpha2048/internal/domain"
)

// AutoPlayConfig は自動プレイの設定
type AutoPlayConfig struct {
	MaxDepth    int
	Delay       time.Duration
	Weights     []float64
	UseAStar    bool
	UseParallel bool
	Verbose     bool
}

// DefaultAutoPlayConfig はデフォルトの設定を返す
func DefaultAutoPlayConfig() AutoPlayConfig {
	return AutoPlayConfig{
		MaxDepth:    4,
		Delay:       100 * time.Millisecond,
		Weights:     nil, // LargestTilePotentialEvaluatorを単独使用するため不要
		UseAStar:    false,
		UseParallel: false,
		Verbose:     true,
	}
}

// AutoPlay は自動でゲームをプレイする
func AutoPlay(w io.Writer, rng *rand.Rand, config AutoPlayConfig) (int, int) {
	game := domain.NewGame(rng)

	// 最大タイルを作ることに特化した単一のポリシーを使用
	evaluator := &domain.LargestTilePotentialEvaluator{}

	var solver interface {
		BestMove(board domain.Board) domain.Direction
	}

	if config.UseAStar {
		solver = domain.NewAStarSolver(evaluator, config.MaxDepth)
	} else if config.UseParallel {
		solver = domain.NewParallelSolver(evaluator, config.MaxDepth)
	} else {
		solver = domain.NewSolver(evaluator, config.MaxDepth)
	}

	moves := 0

	if config.Verbose {
		fmt.Fprintln(w, "=== 2048 AutoPlay ===")
		mode := "Sequential"
		if config.UseAStar {
			mode = "A*"
		} else if config.UseParallel {
			mode = "Parallel"
		}
		fmt.Fprintf(w, "Depth: %d, Mode: %s\n\n", config.MaxDepth, mode)
	}

	for !game.IsGameOver() {
		if config.Verbose {
			fmt.Fprint(w, game.Board())
			fmt.Fprintf(w, "Score: %d, Moves: %d\n", game.Score(), moves)
		}

		dir := solver.BestMove(game.Board())
		if dir == domain.Direction(-1) {
			break
		}

		if config.Verbose {
			fmt.Fprintf(w, "Move: %s\n\n", directionName(dir))
		}

		game.Move(dir)
		moves++

		if config.Delay > 0 {
			time.Sleep(config.Delay)
		}
	}

	// 最終結果は常に表示
	fmt.Fprint(w, game.Board())
	fmt.Fprintln(w, "=== Game Over ===")
	fmt.Fprintf(w, "Final Score: %d\n", game.Score())
	fmt.Fprintf(w, "Total Moves: %d\n", moves)
	fmt.Fprintf(w, "Max Tile: %d\n", maxTile(game.Board()))

	return game.Score(), moves
}

func directionName(dir domain.Direction) string {
	switch dir {
	case domain.Up:
		return "Up"
	case domain.Down:
		return "Down"
	case domain.Left:
		return "Left"
	case domain.Right:
		return "Right"
	default:
		return "Unknown"
	}
}

func maxTile(board domain.Board) int {
	max := 0
	for r := 0; r < 4; r++ {
		for c := 0; c < 4; c++ {
			v := board.Get(r, c)
			if v > max {
				max = v
			}
		}
	}
	return max
}
