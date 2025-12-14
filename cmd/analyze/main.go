package main

import (
	"bufio"
	"flag"
	"fmt"
	"os"
	"strconv"
	"strings"

	"github.com/nnaakkaaii/alpha2048/internal/domain"
)

func main() {
	depth := flag.Int("depth", 5, "initial search depth")
	useBitBoard := flag.Bool("bitboard", false, "use bitboard representation")
	useAStar := flag.Bool("astar", false, "use A* algorithm")
	flag.Parse()

	scanner := bufio.NewScanner(os.Stdin)
	evaluator := domain.NewHeuristicEvaluator()
	
	fmt.Println("=== 2048 Interactive Analyzer ===")
	fmt.Println("Enter board state as 16 numbers (0 for empty), or 'quit' to exit")
	fmt.Println("Example: 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2 2")
	fmt.Println()

	for {
		currentDepth := *depth
		board := inputBoard(scanner)
		if board == nil {
			break
		}

		for {
			fmt.Println("\nCurrent board:")
			fmt.Println(board)

			if board.IsGameOver() {
				fmt.Println("Game Over!")
				break
			}

			fmt.Printf("\nSearch depth: %d\n", currentDepth)
			fmt.Println("Analyzing best move...")
			
			bestMove, analysis := analyzeBestMove(*board, evaluator, currentDepth, *useBitBoard, *useAStar)
			
			if bestMove == domain.Direction(-1) {
				fmt.Println("No valid moves available!")
				break
			}

			fmt.Printf("\n=== Recommended move: %s ===\n", directionToString(bestMove))
			fmt.Println("\nMove scores:")
			for _, dir := range []domain.Direction{domain.Up, domain.Down, domain.Left, domain.Right} {
				score, ok := analysis[dir]
				if ok && score > -1e17 {
					fmt.Printf("  %s: %.2f", directionToString(dir), score)
					if dir == bestMove {
						fmt.Print(" <- BEST")
					}
					fmt.Println()
				}
			}

			fmt.Println("\nOptions:")
			fmt.Println("  1. Apply suggested move and add new tile")
			fmt.Println("  2. Enter custom move and new tile")  
			fmt.Println("  3. Change search depth")
			fmt.Println("  4. New board")
			fmt.Println("  5. Quit")
			fmt.Print("Choice: ")

			scanner.Scan()
			choice := scanner.Text()

			switch choice {
			case "1":
				board = applyMoveWithNewTile(scanner, *board, bestMove)
			case "2":
				board = customMoveWithNewTile(scanner, *board)
			case "3":
				currentDepth = changeDepth(scanner, currentDepth)
			case "4":
				break
			case "5":
				return
			default:
				fmt.Println("Invalid choice")
			}

			if choice == "4" {
				break
			}
		}
	}
}

func inputBoard(scanner *bufio.Scanner) *domain.Board {
	fmt.Println("Enter board (16 numbers separated by spaces, or 'quit'):")
	scanner.Scan()
	input := scanner.Text()
	
	if input == "quit" {
		return nil
	}

	parts := strings.Fields(input)
	if len(parts) != 16 {
		fmt.Println("Error: need exactly 16 numbers")
		return inputBoard(scanner)
	}

	var cells [4][4]int
	for i := 0; i < 16; i++ {
		val, err := strconv.Atoi(parts[i])
		if err != nil {
			fmt.Printf("Error parsing number: %v\n", err)
			return inputBoard(scanner)
		}
		cells[i/4][i%4] = val
	}

	board := domain.NewBoardFromCells(cells)
	return &board
}

func analyzeBestMove(board domain.Board, evaluator domain.Evaluator, depth int, useBitBoard, useAStar bool) (domain.Direction, map[domain.Direction]float64) {
	analysis := make(map[domain.Direction]float64)

	if useBitBoard {
		// BitBoard版のソルバーを使用
		if useAStar {
			// BitBoard版のA*は未実装のため通常版を使用
			solver := domain.NewAStarSolver(evaluator, depth)
			bestMove := solver.BestMove(board)
			
			// 各方向のスコアを計算
			for _, dir := range []domain.Direction{domain.Up, domain.Down, domain.Left, domain.Right} {
				newBoard, _ := board.SwipeWithoutSpawn(dir)
				if newBoard.Equal(board) {
					analysis[dir] = -1e18
					continue
				}
				analysis[dir] = evaluator.Evaluate(newBoard)
			}
			return bestMove, analysis
		} else {
			// BitBoardを使ったExpectimax
			bb := domain.NewBitBoard(board)
			bestDir := domain.Direction(-1)
			bestScore := float64(-1e18)

			// 各方向を評価
			for _, dir := range []domain.Direction{domain.Up, domain.Down, domain.Left, domain.Right} {
				newBB, _ := bb.Swipe(dir)
				if newBB.Equal(bb) {
					analysis[dir] = -1e18
					continue
				}
				
				solver := domain.NewBitBoardSolver(evaluator, depth)
				score := expectScoreBitBoard(solver, newBB, depth-1, evaluator)
				
				analysis[dir] = score
				if score > bestScore {
					bestScore = score
					bestDir = dir
				}
			}
			
			return bestDir, analysis
		}
	} else {
		// 通常のBoard表現
		var solver interface {
			BestMove(domain.Board) domain.Direction
		}
		
		if useAStar {
			solver = domain.NewAStarSolver(evaluator, depth)
		} else {
			solver = domain.NewSolver(evaluator, depth)
		}
		
		bestMove := solver.BestMove(board)
		
		// 各方向のスコアを計算
		for _, dir := range []domain.Direction{domain.Up, domain.Down, domain.Left, domain.Right} {
			newBoard, _ := board.SwipeWithoutSpawn(dir)
			if newBoard.Equal(board) {
				analysis[dir] = -1e18
				continue
			}
			
			if useAStar {
				analysis[dir] = evaluator.Evaluate(newBoard)
			} else {
				s := domain.NewSolver(evaluator, depth)
				analysis[dir] = expectScore(s, newBoard, depth-1, evaluator)
			}
		}
		
		return bestMove, analysis
	}
}

func expectScore(solver *domain.Solver, board domain.Board, depth int, evaluator domain.Evaluator) float64 {
	emptyCells := board.EmptyCells()
	if len(emptyCells) == 0 || depth <= 0 {
		return evaluator.Evaluate(board)
	}

	sampleCells := emptyCells
	maxSample := 4
	if len(emptyCells) > maxSample {
		sampleCells = make([][2]int, 0, maxSample)
		step := len(emptyCells) / maxSample
		if step == 0 {
			step = 1
		}
		for i := 0; i < len(emptyCells) && len(sampleCells) < maxSample; i += step {
			sampleCells = append(sampleCells, emptyCells[i])
		}
	}

	totalScore := 0.0
	for _, pos := range sampleCells {
		board2 := board.Set(pos[0], pos[1], 2)
		score2 := searchMax(board2, depth, evaluator)
		
		board4 := board.Set(pos[0], pos[1], 4)
		score4 := searchMax(board4, depth, evaluator)
		
		totalScore += 0.9*score2 + 0.1*score4
	}

	return totalScore / float64(len(sampleCells))
}

func expectScoreBitBoard(solver *domain.BitBoardSolver, bb domain.BitBoard, depth int, evaluator domain.Evaluator) float64 {
	emptyCells := bb.EmptyCells()
	if len(emptyCells) == 0 || depth <= 0 {
		return evaluator.Evaluate(bb.ToBoard())
	}

	sampleCells := emptyCells
	maxSample := 4
	if len(emptyCells) > maxSample {
		sampleCells = make([][2]int, 0, maxSample)
		step := len(emptyCells) / maxSample
		if step == 0 {
			step = 1
		}
		for i := 0; i < len(emptyCells) && len(sampleCells) < maxSample; i += step {
			sampleCells = append(sampleCells, emptyCells[i])
		}
	}

	totalScore := 0.0
	for _, pos := range sampleCells {
		bb2 := bb.Set(pos[0], pos[1], 2)
		score2 := searchMaxBitBoard(bb2, depth, evaluator)
		
		bb4 := bb.Set(pos[0], pos[1], 4)
		score4 := searchMaxBitBoard(bb4, depth, evaluator)
		
		totalScore += 0.9*score2 + 0.1*score4
	}

	return totalScore / float64(len(sampleCells))
}

func searchMax(board domain.Board, depth int, evaluator domain.Evaluator) float64 {
	if depth <= 0 || board.IsGameOver() {
		return evaluator.Evaluate(board)
	}

	bestScore := float64(-1e18)
	hasMoved := false

	for _, dir := range []domain.Direction{domain.Up, domain.Down, domain.Left, domain.Right} {
		newBoard, _ := board.SwipeWithoutSpawn(dir)
		if newBoard.Equal(board) {
			continue
		}
		hasMoved = true

		solver := domain.NewSolver(evaluator, depth)
		score := expectScore(solver, newBoard, depth-1, evaluator)
		if score > bestScore {
			bestScore = score
		}
	}

	if !hasMoved {
		return evaluator.Evaluate(board)
	}

	return bestScore
}

func searchMaxBitBoard(bb domain.BitBoard, depth int, evaluator domain.Evaluator) float64 {
	if depth <= 0 || bb.IsGameOver() {
		return evaluator.Evaluate(bb.ToBoard())
	}

	bestScore := float64(-1e18)
	hasMoved := false

	for _, dir := range []domain.Direction{domain.Up, domain.Down, domain.Left, domain.Right} {
		newBB, _ := bb.Swipe(dir)
		if newBB.Equal(bb) {
			continue
		}
		hasMoved = true

		solver := domain.NewBitBoardSolver(evaluator, depth)
		score := expectScoreBitBoard(solver, newBB, depth-1, evaluator)
		if score > bestScore {
			bestScore = score
		}
	}

	if !hasMoved {
		return evaluator.Evaluate(bb.ToBoard())
	}

	return bestScore
}

func applyMoveWithNewTile(scanner *bufio.Scanner, board domain.Board, dir domain.Direction) *domain.Board {
	newBoard, score := board.SwipeWithoutSpawn(dir)
	fmt.Printf("\nApplied %s (score gained: +%d)\n", directionToString(dir), score)
	fmt.Println(newBoard)
	
	fmt.Println("\nEmpty cells:")
	emptyCells := newBoard.EmptyCells()
	for i, cell := range emptyCells {
		fmt.Printf("  %d: (%d,%d)\n", i, cell[0], cell[1])
	}
	
	fmt.Print("\nEnter new tile position (row col) and value (2 or 4): ")
	scanner.Scan()
	parts := strings.Fields(scanner.Text())
	
	if len(parts) != 3 {
		fmt.Println("Invalid input. Format: row col value")
		return &board
	}
	
	row, _ := strconv.Atoi(parts[0])
	col, _ := strconv.Atoi(parts[1])
	val, _ := strconv.Atoi(parts[2])
	
	if row < 0 || row > 3 || col < 0 || col > 3 {
		fmt.Println("Invalid position")
		return &board
	}
	
	if val != 2 && val != 4 {
		fmt.Println("Value must be 2 or 4")
		return &board
	}
	
	finalBoard := newBoard.Set(row, col, val)
	return &finalBoard
}

func customMoveWithNewTile(scanner *bufio.Scanner, board domain.Board) *domain.Board {
	fmt.Print("Enter direction (u/d/l/r): ")
	scanner.Scan()
	dirStr := scanner.Text()
	
	var dir domain.Direction
	switch dirStr {
	case "u":
		dir = domain.Up
	case "d":
		dir = domain.Down
	case "l":
		dir = domain.Left
	case "r":
		dir = domain.Right
	default:
		fmt.Println("Invalid direction")
		return &board
	}
	
	return applyMoveWithNewTile(scanner, board, dir)
}

func changeDepth(scanner *bufio.Scanner, currentDepth int) int {
	fmt.Printf("Enter new depth (current: %d): ", currentDepth)
	scanner.Scan()
	newDepth, err := strconv.Atoi(scanner.Text())
	if err != nil || newDepth < 1 || newDepth > 10 {
		fmt.Println("Invalid depth (must be 1-10)")
		return currentDepth
	}
	return newDepth
}

func directionToString(dir domain.Direction) string {
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