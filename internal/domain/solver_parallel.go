package domain

import (
	"runtime"
	"sync"
)

// ParallelSolver は並列処理を最適化したソルバー
type ParallelSolver struct {
	evaluator Evaluator
	maxDepth  int
	workers   int
}

// NewParallelSolver は新しいParallelSolverを生成する
func NewParallelSolver(evaluator Evaluator, maxDepth int) *ParallelSolver {
	return &ParallelSolver{
		evaluator: evaluator,
		maxDepth:  maxDepth,
		workers:   runtime.NumCPU(),
	}
}

// BestMove は現在の盤面から最良の手を返す（トップレベルのみ並列化）
func (s *ParallelSolver) BestMove(board Board) Direction {
	type result struct {
		dir   Direction
		score float64
	}
	
	// 有効な手を事前にフィルタリング
	validMoves := make([]Direction, 0, 4)
	for _, dir := range []Direction{Up, Down, Left, Right} {
		newBoard, _ := board.SwipeWithoutSpawn(dir)
		if !newBoard.Equal(board) {
			validMoves = append(validMoves, dir)
		}
	}
	
	if len(validMoves) == 0 {
		return Direction(-1)
	}
	
	// 1手しかない場合は並列化不要
	if len(validMoves) == 1 {
		return validMoves[0]
	}
	
	// 並列に評価
	results := make([]result, len(validMoves))
	var wg sync.WaitGroup
	
	for i, dir := range validMoves {
		wg.Add(1)
		go func(idx int, d Direction) {
			defer wg.Done()
			newBoard, _ := board.SwipeWithoutSpawn(d)
			score := s.expectedScore(newBoard, s.maxDepth-1)
			results[idx] = result{dir: d, score: score}
		}(i, dir)
	}
	
	wg.Wait()
	
	// 最良の手を選択
	bestDir := results[0].dir
	bestScore := results[0].score
	
	for i := 1; i < len(results); i++ {
		if results[i].score > bestScore {
			bestScore = results[i].score
			bestDir = results[i].dir
		}
	}
	
	return bestDir
}

// expectedScore はスポーンの期待値を計算する（順次実行）
func (s *ParallelSolver) expectedScore(board Board, depth int) float64 {
	emptyCells := board.EmptyCells()
	if len(emptyCells) == 0 || depth <= 0 {
		return s.evaluator.Evaluate(board)
	}
	
	// サンプリング
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
	
	// 各空きマスにスポーンした場合の期待値
	totalScore := 0.0
	for _, pos := range sampleCells {
		board2 := board.Set(pos[0], pos[1], 2)
		score2 := s.searchMax(board2, depth)
		
		board4 := board.Set(pos[0], pos[1], 4)
		score4 := s.searchMax(board4, depth)
		
		totalScore += spawn2Prob*score2 + spawn4Prob*score4
	}
	
	return totalScore / float64(len(sampleCells))
}

// searchMax はプレイヤーの最善手を探索（順次実行）
func (s *ParallelSolver) searchMax(board Board, depth int) float64 {
	if depth <= 0 || board.IsGameOver() {
		return s.evaluator.Evaluate(board)
	}
	
	bestScore := float64(-1e18)
	hasMoved := false
	
	for _, dir := range []Direction{Up, Down, Left, Right} {
		newBoard, _ := board.SwipeWithoutSpawn(dir)
		if newBoard.Equal(board) {
			continue
		}
		hasMoved = true
		
		score := s.expectedScore(newBoard, depth-1)
		if score > bestScore {
			bestScore = score
		}
	}
	
	if !hasMoved {
		return s.evaluator.Evaluate(board)
	}
	
	return bestScore
}