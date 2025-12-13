package domain

import (
	"math/bits"
)

// BitBoard は2048の盤面を64ビット整数で表現
// 各タイルは4ビットで表現（0-15の指数: 0=空, 1=2, 2=4, 3=8, ..., 15=32768）
// 16個のタイル × 4ビット = 64ビット
type BitBoard uint64

// NewBitBoard は通常のBoardからBitBoardを生成
func NewBitBoard(b Board) BitBoard {
	var bb BitBoard
	for r := 0; r < 4; r++ {
		for c := 0; c < 4; c++ {
			val := b.Get(r, c)
			if val > 0 {
				// 2の何乗かを計算（2→1, 4→2, 8→3, ...）
				exp := bits.TrailingZeros(uint(val))
				bb.setTile(r, c, exp)
			}
		}
	}
	return bb
}

// ToBoard はBitBoardを通常のBoardに変換
func (bb BitBoard) ToBoard() Board {
	var cells [4][4]int
	for r := 0; r < 4; r++ {
		for c := 0; c < 4; c++ {
			exp := bb.getTile(r, c)
			if exp > 0 {
				cells[r][c] = 1 << exp // 2^exp
			}
		}
	}
	return NewBoardFromCells(cells)
}

// getTile は指定位置のタイル値（指数）を取得
func (bb BitBoard) getTile(row, col int) int {
	shift := (row*4 + col) * 4
	return int((bb >> shift) & 0xF)
}

// setTile は指定位置にタイル値（指数）を設定
func (bb *BitBoard) setTile(row, col, exp int) {
	shift := (row*4 + col) * 4
	mask := ^(BitBoard(0xF) << shift)
	*bb = (*bb & mask) | (BitBoard(exp) << shift)
}

// SwipeLeft は左スワイプを高速実行
func (bb BitBoard) SwipeLeft() (BitBoard, int) {
	result := BitBoard(0)
	score := 0
	
	for r := 0; r < 4; r++ {
		row := bb.extractRow(r)
		newRow, rowScore := slideRow(row)
		result = result.setRow(r, newRow)
		score += rowScore
	}
	
	return result, score
}

// SwipeRight は右スワイプを高速実行
func (bb BitBoard) SwipeRight() (BitBoard, int) {
	result := BitBoard(0)
	score := 0
	
	for r := 0; r < 4; r++ {
		row := bb.extractRow(r)
		reversed := reverseRow(row)
		newRow, rowScore := slideRow(reversed)
		result = result.setRow(r, reverseRow(newRow))
		score += rowScore
	}
	
	return result, score
}

// SwipeUp は上スワイプを高速実行
func (bb BitBoard) SwipeUp() (BitBoard, int) {
	// 転置してから左スワイプ、その後再転置
	transposed := bb.transpose()
	result, score := transposed.SwipeLeft()
	return result.transpose(), score
}

// SwipeDown は下スワイプを高速実行
func (bb BitBoard) SwipeDown() (BitBoard, int) {
	// 転置してから右スワイプ、その後再転置
	transposed := bb.transpose()
	result, score := transposed.SwipeRight()
	return result.transpose(), score
}

// extractRow は指定行を16ビット値として抽出
func (bb BitBoard) extractRow(row int) uint16 {
	shift := row * 16
	return uint16((bb >> shift) & 0xFFFF)
}

// setRow は指定行に16ビット値を設定
func (bb BitBoard) setRow(row int, rowVal uint16) BitBoard {
	shift := row * 16
	mask := ^(BitBoard(0xFFFF) << shift)
	return (bb & mask) | (BitBoard(rowVal) << shift)
}

// slideRow は1行を左にスライドしてマージ
func slideRow(row uint16) (uint16, int) {
	// 4つのタイルを抽出
	tiles := [4]int{
		int(row & 0xF),
		int((row >> 4) & 0xF),
		int((row >> 8) & 0xF),
		int((row >> 12) & 0xF),
	}
	
	score := 0
	result := [4]int{}
	writePos := 0
	
	// 左にスライドしてマージ
	for readPos := 0; readPos < 4; readPos++ {
		if tiles[readPos] == 0 {
			continue
		}
		
		if writePos > 0 && result[writePos-1] == tiles[readPos] && result[writePos-1] < 15 {
			// マージ可能
			result[writePos-1]++
			score += 1 << result[writePos-1] // 2^(exp+1)がスコア
		} else {
			result[writePos] = tiles[readPos]
			writePos++
		}
	}
	
	// 16ビット値に戻す
	newRow := uint16(0)
	for i := 0; i < 4; i++ {
		newRow |= uint16(result[i]) << (i * 4)
	}
	
	return newRow, score
}

// reverseRow は行を反転
func reverseRow(row uint16) uint16 {
	return ((row & 0xF) << 12) |
		   ((row & 0xF0) << 4) |
		   ((row & 0xF00) >> 4) |
		   ((row & 0xF000) >> 12)
}

// transpose は盤面を転置
func (bb BitBoard) transpose() BitBoard {
	var result BitBoard
	for r := 0; r < 4; r++ {
		for c := 0; c < 4; c++ {
			val := bb.getTile(r, c)
			result.setTile(c, r, val)
		}
	}
	return result
}

// EmptyCells は空きマスの位置を返す
func (bb BitBoard) EmptyCells() [][2]int {
	cells := make([][2]int, 0, 16)
	for r := 0; r < 4; r++ {
		for c := 0; c < 4; c++ {
			if bb.getTile(r, c) == 0 {
				cells = append(cells, [2]int{r, c})
			}
		}
	}
	return cells
}

// Equal は2つのBitBoardが等しいか判定
func (bb BitBoard) Equal(other BitBoard) bool {
	return bb == other
}

// IsGameOver はゲームオーバーか判定
func (bb BitBoard) IsGameOver() bool {
	// 空きマスがあるか
	if len(bb.EmptyCells()) > 0 {
		return false
	}
	
	// 各方向にスワイプして変化があるか
	for _, dir := range []Direction{Up, Down, Left, Right} {
		newBB, _ := bb.Swipe(dir)
		if !newBB.Equal(bb) {
			return false
		}
	}
	
	return true
}

// Swipe は指定方向にスワイプ
func (bb BitBoard) Swipe(dir Direction) (BitBoard, int) {
	switch dir {
	case Up:
		return bb.SwipeUp()
	case Down:
		return bb.SwipeDown()
	case Left:
		return bb.SwipeLeft()
	case Right:
		return bb.SwipeRight()
	default:
		return bb, 0
	}
}

// Set はタイルを設定（値は2の累乗）
func (bb BitBoard) Set(row, col, value int) BitBoard {
	newBB := bb
	if value == 0 {
		newBB.setTile(row, col, 0)
	} else {
		exp := bits.TrailingZeros(uint(value))
		newBB.setTile(row, col, exp)
	}
	return newBB
}

// Get はタイルの値を取得
func (bb BitBoard) Get(row, col int) int {
	exp := bb.getTile(row, col)
	if exp == 0 {
		return 0
	}
	return 1 << exp
}