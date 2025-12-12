package usecase

import (
	"bufio"
	"fmt"
	"io"
	"math/rand"
	"strings"

	"github.com/nnaakkaaii/alpha2048/internal/domain"
)

// PlayGame はCLIで2048ゲームを実行する
func PlayGame(r io.Reader, w io.Writer, rng *rand.Rand) {
	game := domain.NewGame(rng)
	reader := bufio.NewReader(r)

	fmt.Fprintln(w, "=== 2048 ===")
	fmt.Fprintln(w, "Controls: w=Up, s=Down, a=Left, d=Right, q=Quit")
	fmt.Fprintln(w)

	for {
		fmt.Fprint(w, game.Board())
		fmt.Fprintf(w, "Score: %d\n", game.Score())

		if game.IsGameOver() {
			fmt.Fprintln(w, "Game Over!")
			break
		}

		fmt.Fprint(w, "Move: ")
		input, err := reader.ReadString('\n')
		if err != nil {
			break
		}

		input = strings.TrimSpace(strings.ToLower(input))
		if input == "q" {
			fmt.Fprintln(w, "Quit.")
			break
		}

		dir, ok := parseDirection(input)
		if !ok {
			fmt.Fprintln(w, "Invalid input. Use w/a/s/d or q to quit.")
			continue
		}

		if !game.Move(dir) {
			fmt.Fprintln(w, "Cannot move in that direction.")
		}
		fmt.Fprintln(w)
	}
}

func parseDirection(input string) (domain.Direction, bool) {
	switch input {
	case "w":
		return domain.Up, true
	case "s":
		return domain.Down, true
	case "a":
		return domain.Left, true
	case "d":
		return domain.Right, true
	default:
		return 0, false
	}
}
