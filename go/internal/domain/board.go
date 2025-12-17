package domain

import "fmt"

// Direction はスワイプの方向を表す
type Direction int

const (
	Up Direction = iota
	Down
	Left
	Right
)

// SpawnValues はスワイプ後に空きマスに出現しうる値
var SpawnValues = []int{2, 4}

// Board は4x4の2048ゲーム盤面を表す（immutable）
type Board struct {
	cells [4][4]int
}

// NewBoard は空のBoardを生成する
func NewBoard() Board {
	return Board{}
}

// NewBoardFromCells はセルの値を指定してBoardを生成する
func NewBoardFromCells(cells [4][4]int) Board {
	return Board{cells: cells}
}

// Get は指定した位置のセル値を取得する
func (b Board) Get(row, col int) int {
	return b.cells[row][col]
}

// Set は指定した位置に値を設定した新しいBoardを返す
func (b Board) Set(row, col, value int) Board {
	newBoard := b.Copy()
	newBoard.cells[row][col] = value
	return newBoard
}

// Copy はBoardのコピーを返す
func (b Board) Copy() Board {
	var newCells [4][4]int
	for r := 0; r < 4; r++ {
		for c := 0; c < 4; c++ {
			newCells[r][c] = b.cells[r][c]
		}
	}
	return Board{cells: newCells}
}

// EmptyCells は空のセルの座標一覧を返す
func (b Board) EmptyCells() [][2]int {
	var empty [][2]int
	for r := 0; r < 4; r++ {
		for c := 0; c < 4; c++ {
			if b.cells[r][c] == 0 {
				empty = append(empty, [2]int{r, c})
			}
		}
	}
	return empty
}

// Swipe は指定した方向にスワイプし、取りうる全ての盤面を返す
// 空きマスに2または4がspawnする全ての可能性を列挙する
// スワイプで盤面が変化しない場合は空のスライスを返す
func (b Board) Swipe(dir Direction) []Board {
	swiped, _ := b.SwipeWithoutSpawn(dir)

	// 盤面が変化しない場合は空を返す
	if swiped.Equal(b) {
		return nil
	}

	// 空きマスを取得
	emptyCells := swiped.EmptyCells()
	if len(emptyCells) == 0 {
		return nil
	}

	// 各空きマスにSpawnValuesの値を配置した全ての盤面を生成
	results := make([]Board, 0, len(emptyCells)*len(SpawnValues))

	for _, pos := range emptyCells {
		for _, val := range SpawnValues {
			newBoard := swiped.Set(pos[0], pos[1], val)
			results = append(results, newBoard)
		}
	}

	return results
}

// SwipeWithoutSpawn は指定した方向にスワイプした結果の盤面とスコアを返す（spawnなし）
func (b Board) SwipeWithoutSpawn(dir Direction) (Board, int) {
	newCells := [4][4]int{}
	totalScore := 0

	switch dir {
	case Left:
		for r := 0; r < 4; r++ {
			row := b.getRow(r)
			merged, score := mergeLine(row)
			totalScore += score
			for c := 0; c < 4; c++ {
				newCells[r][c] = merged[c]
			}
		}
	case Right:
		for r := 0; r < 4; r++ {
			row := b.getRow(r)
			reversed := reverseLine(row)
			merged, score := mergeLine(reversed)
			totalScore += score
			result := reverseLine(merged)
			for c := 0; c < 4; c++ {
				newCells[r][c] = result[c]
			}
		}
	case Up:
		for c := 0; c < 4; c++ {
			col := b.getCol(c)
			merged, score := mergeLine(col)
			totalScore += score
			for r := 0; r < 4; r++ {
				newCells[r][c] = merged[r]
			}
		}
	case Down:
		for c := 0; c < 4; c++ {
			col := b.getCol(c)
			reversed := reverseLine(col)
			merged, score := mergeLine(reversed)
			totalScore += score
			result := reverseLine(merged)
			for r := 0; r < 4; r++ {
				newCells[r][c] = result[r]
			}
		}
	}

	return Board{cells: newCells}, totalScore
}

// getRow は指定した行を配列として返す
func (b Board) getRow(row int) [4]int {
	return b.cells[row]
}

// getCol は指定した列を配列として返す
func (b Board) getCol(col int) [4]int {
	var result [4]int
	for r := 0; r < 4; r++ {
		result[r] = b.cells[r][col]
	}
	return result
}

// mergeLine は1行/1列を左方向にマージし、結果とスコアを返す
func mergeLine(line [4]int) ([4]int, int) {
	score := 0

	// 0を除去して詰める
	var nonZero []int
	for _, v := range line {
		if v != 0 {
			nonZero = append(nonZero, v)
		}
	}

	// 同じ値が隣接していたらマージ
	var merged []int
	for i := 0; i < len(nonZero); i++ {
		if i+1 < len(nonZero) && nonZero[i] == nonZero[i+1] {
			newVal := nonZero[i] * 2
			merged = append(merged, newVal)
			score += newVal
			i++ // 次の要素をスキップ
		} else {
			merged = append(merged, nonZero[i])
		}
	}

	// 4要素の配列に戻す
	var result [4]int
	for i, v := range merged {
		if i < 4 {
			result[i] = v
		}
	}
	return result, score
}

// reverseLine は配列を反転する
func reverseLine(line [4]int) [4]int {
	return [4]int{line[3], line[2], line[1], line[0]}
}

// IsGameOver は全方向にスワイプできない（ゲームオーバー）かどうかを返す
func (b Board) IsGameOver() bool {
	for _, dir := range []Direction{Up, Down, Left, Right} {
		swiped, _ := b.SwipeWithoutSpawn(dir)
		if !swiped.Equal(b) {
			return false
		}
	}
	return true
}

// Equal は2つのBoardが等しいかどうかを返す
func (b Board) Equal(other Board) bool {
	return b.cells == other.cells
}

// String はBoardをASCIIアートとして表示する
func (b Board) String() string {
	line := "+------+------+------+------+"
	result := line + "\n"
	for r := 0; r < 4; r++ {
		result += "|"
		for c := 0; c < 4; c++ {
			if b.cells[r][c] == 0 {
				result += "      |"
			} else {
				result += fmt.Sprintf("%5d |", b.cells[r][c])
			}
		}
		result += "\n" + line + "\n"
	}
	return result
}
