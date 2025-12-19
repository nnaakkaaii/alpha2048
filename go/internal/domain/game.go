package domain

import "math/rand"

// Game は2048ゲームの状態を管理する
type Game struct {
	board Board
	score int
	rng   *rand.Rand
}

// NewGame は新しいゲームを開始する
func NewGame(rng *rand.Rand) *Game {
	g := &Game{
		board: NewBoard(),
		score: 0,
		rng:   rng,
	}
	// 初期配置として2つのタイルを配置
	g.spawnTile()
	g.spawnTile()
	return g
}

// Board は現在の盤面を返す
func (g *Game) Board() Board {
	return g.board
}

// Score は現在のスコアを返す
func (g *Game) Score() int {
	return g.score
}

// IsGameOver はゲームオーバーかどうかを返す
func (g *Game) IsGameOver() bool {
	return g.board.IsGameOver()
}

// Move は指定した方向にスワイプを実行する
// 盤面が変化した場合はtrueを返す
func (g *Game) Move(dir Direction) bool {
	newBoard, score := g.board.SwipeWithoutSpawn(dir)

	if newBoard.Equal(g.board) {
		return false
	}

	g.score += score
	g.board = newBoard
	g.spawnTile()
	return true
}

// spawnTile は空きマスにランダムにタイルを配置する
func (g *Game) spawnTile() {
	empty := g.board.EmptyCells()
	if len(empty) == 0 {
		return
	}

	pos := empty[g.rng.Intn(len(empty))]
	val := SpawnValues[g.rng.Intn(len(SpawnValues))]
	g.board = g.board.Set(pos[0], pos[1], val)
}
