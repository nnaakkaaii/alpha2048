package domain

import (
	"container/heap"
)

// スポーン確率（2が90%、4が10%）
const (
	spawn2Prob = 0.9
	spawn4Prob = 0.1
)

// Solver はExpectimaxアルゴリズムで最良の手を探索する
type Solver struct {
	evaluator Evaluator
	maxDepth  int
}

// NewSolver は新しいSolverを生成する
func NewSolver(evaluator Evaluator, maxDepth int) *Solver {
	return &Solver{
		evaluator: evaluator,
		maxDepth:  maxDepth,
	}
}

// BestMove は現在の盤面から最良の手を返す
// 有効な手がない場合は-1を返す
func (s *Solver) BestMove(board Board) Direction {
	bestDir := Direction(-1)
	bestScore := float64(-1e18)

	for _, dir := range []Direction{Up, Down, Left, Right} {
		newBoard, _ := board.SwipeWithoutSpawn(dir)
		if newBoard.Equal(board) {
			continue
		}

		// スポーン後の期待値を計算
		score := s.expectedScore(newBoard, s.maxDepth-1)

		if score > bestScore {
			bestScore = score
			bestDir = dir
		}
	}

	return bestDir
}

// expectedScore はスポーンの期待値を計算する
func (s *Solver) expectedScore(board Board, depth int) float64 {
	emptyCells := board.EmptyCells()
	if len(emptyCells) == 0 || depth <= 0 {
		return s.evaluator.Evaluate(board)
	}

	// 空きマスが多い場合はサンプリング（高速化）
	sampleCells := emptyCells
	maxSample := 6
	if len(emptyCells) > maxSample {
		// 戦略的にサンプリング: 均等に分散して選択
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
		board2 := board.Set(pos[0], pos[1], 2)
		score2 := s.searchMax(board2, depth)

		// 4がスポーンした場合
		board4 := board.Set(pos[0], pos[1], 4)
		score4 := s.searchMax(board4, depth)

		totalScore += spawn2Prob*score2 + spawn4Prob*score4
	}

	return totalScore / float64(len(sampleCells))
}

// searchMax はプレイヤーの最善手を探索
func (s *Solver) searchMax(board Board, depth int) float64 {
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

// AStarSolver はA*アルゴリズムで探索する（優先度付きキュー使用）
type AStarSolver struct {
	evaluator Evaluator
	maxDepth  int
}

// NewAStarSolver は新しいAStarSolverを生成する
func NewAStarSolver(evaluator Evaluator, maxDepth int) *AStarSolver {
	return &AStarSolver{
		evaluator: evaluator,
		maxDepth:  maxDepth,
	}
}

// searchNode はA*探索のノード
type searchNode struct {
	board    Board
	depth    int
	g        float64 // コスト（手数）
	h        float64 // ヒューリスティック（評価値の負）
	firstDir Direction
	index    int
}

// f値を返す
func (n *searchNode) f() float64 {
	return n.g + n.h
}

// priorityQueue は優先度付きキュー
type priorityQueue []*searchNode

func (pq priorityQueue) Len() int { return len(pq) }

func (pq priorityQueue) Less(i, j int) bool {
	return pq[i].f() < pq[j].f()
}

func (pq priorityQueue) Swap(i, j int) {
	pq[i], pq[j] = pq[j], pq[i]
	pq[i].index = i
	pq[j].index = j
}

func (pq *priorityQueue) Push(x interface{}) {
	n := len(*pq)
	node := x.(*searchNode)
	node.index = n
	*pq = append(*pq, node)
}

func (pq *priorityQueue) Pop() interface{} {
	old := *pq
	n := len(old)
	node := old[n-1]
	old[n-1] = nil
	node.index = -1
	*pq = old[0 : n-1]
	return node
}

// BestMove はA*アルゴリズムで最良の手を返す
func (s *AStarSolver) BestMove(board Board) Direction {
	pq := make(priorityQueue, 0)
	heap.Init(&pq)

	// 初期ノードを追加（各方向、スポーンなし）
	for _, dir := range []Direction{Up, Down, Left, Right} {
		newBoard, _ := board.SwipeWithoutSpawn(dir)
		if newBoard.Equal(board) {
			continue
		}
		h := -s.evaluator.Evaluate(newBoard)
		node := &searchNode{
			board:    newBoard,
			depth:    1,
			g:        1.0,
			h:        h,
			firstDir: dir,
		}
		heap.Push(&pq, node)
	}

	if pq.Len() == 0 {
		return Direction(-1)
	}

	bestDir := Direction(-1)
	bestScore := float64(1e18)
	explored := 0
	maxExplore := 500

	for pq.Len() > 0 && explored < maxExplore {
		node := heap.Pop(&pq).(*searchNode)
		explored++

		if node.depth >= s.maxDepth {
			if node.f() < bestScore {
				bestScore = node.f()
				bestDir = node.firstDir
			}
			continue
		}

		// 子ノードを展開（スポーンなしで次の手を探索）
		for _, dir := range []Direction{Up, Down, Left, Right} {
			newBoard, _ := node.board.SwipeWithoutSpawn(dir)
			if newBoard.Equal(node.board) {
				continue
			}
			h := -s.evaluator.Evaluate(newBoard)
			child := &searchNode{
				board:    newBoard,
				depth:    node.depth + 1,
				g:        node.g + 1.0,
				h:        h,
				firstDir: node.firstDir,
			}
			heap.Push(&pq, child)
		}
	}

	return bestDir
}
