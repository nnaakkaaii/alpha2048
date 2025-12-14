package domain

// BitBoardSolver はビットボード表現を使った高速ソルバー
type BitBoardSolver struct {
	evaluator Evaluator
	maxDepth  int
}

// NewBitBoardSolver は新しいBitBoardSolverを生成する
func NewBitBoardSolver(evaluator Evaluator, maxDepth int) *BitBoardSolver {
	return &BitBoardSolver{
		evaluator: evaluator,
		maxDepth:  maxDepth,
	}
}

// BestMove は現在の盤面から最良の手を返す
func (s *BitBoardSolver) BestMove(board Board) Direction {
	// 通常のBoardをBitBoardに変換
	bb := NewBitBoard(board)
	
	bestDir := Direction(-1)
	bestScore := float64(-1e18)
	
	for _, dir := range []Direction{Up, Down, Left, Right} {
		newBB, _ := bb.Swipe(dir)
		if newBB.Equal(bb) {
			continue
		}
		
		// スポーン後の期待値を計算
		score := s.expectedScore(newBB, s.maxDepth-1)
		
		if score > bestScore {
			bestScore = score
			bestDir = dir
		}
	}
	
	return bestDir
}

// expectedScore はスポーンの期待値を計算する
func (s *BitBoardSolver) expectedScore(bb BitBoard, depth int) float64 {
	emptyCells := bb.EmptyCells()
	if len(emptyCells) == 0 || depth <= 0 {
		// BitBoardを通常のBoardに変換して評価
		return s.evaluator.Evaluate(bb.ToBoard())
	}
	
	// 空きマスが多い場合はサンプリング（高速化）
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
		// 2がスポーンした場合
		bb2 := bb.Set(pos[0], pos[1], 2)
		score2 := s.searchMax(bb2, depth)
		
		// 4がスポーンした場合
		bb4 := bb.Set(pos[0], pos[1], 4)
		score4 := s.searchMax(bb4, depth)
		
		totalScore += spawn2Prob*score2 + spawn4Prob*score4
	}
	
	return totalScore / float64(len(sampleCells))
}

// searchMax はプレイヤーの最善手を探索
func (s *BitBoardSolver) searchMax(bb BitBoard, depth int) float64 {
	if depth <= 0 || bb.IsGameOver() {
		return s.evaluator.Evaluate(bb.ToBoard())
	}
	
	bestScore := float64(-1e18)
	hasMoved := false
	
	for _, dir := range []Direction{Up, Down, Left, Right} {
		newBB, _ := bb.Swipe(dir)
		if newBB.Equal(bb) {
			continue
		}
		hasMoved = true
		
		score := s.expectedScore(newBB, depth-1)
		if score > bestScore {
			bestScore = score
		}
	}
	
	if !hasMoved {
		return s.evaluator.Evaluate(bb.ToBoard())
	}
	
	return bestScore
}