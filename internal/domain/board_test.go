package domain

import (
	"strings"
	"testing"
)

func TestMergeLine(t *testing.T) {
	tests := []struct {
		name     string
		input    [4]int
		expected [4]int
	}{
		{
			name:     "empty line",
			input:    [4]int{0, 0, 0, 0},
			expected: [4]int{0, 0, 0, 0},
		},
		{
			name:     "no merge needed",
			input:    [4]int{2, 4, 8, 16},
			expected: [4]int{2, 4, 8, 16},
		},
		{
			name:     "simple merge",
			input:    [4]int{2, 2, 0, 0},
			expected: [4]int{4, 0, 0, 0},
		},
		{
			name:     "merge with gap",
			input:    [4]int{2, 0, 2, 0},
			expected: [4]int{4, 0, 0, 0},
		},
		{
			name:     "two merges",
			input:    [4]int{2, 2, 4, 4},
			expected: [4]int{4, 8, 0, 0},
		},
		{
			name:     "chain does not cascade",
			input:    [4]int{2, 2, 2, 2},
			expected: [4]int{4, 4, 0, 0},
		},
		{
			name:     "three same values",
			input:    [4]int{2, 2, 2, 0},
			expected: [4]int{4, 2, 0, 0},
		},
		{
			name:     "shift left",
			input:    [4]int{0, 0, 0, 2},
			expected: [4]int{2, 0, 0, 0},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, _ := mergeLine(tt.input)
			if result != tt.expected {
				t.Errorf("mergeLine(%v) = %v, want %v", tt.input, result, tt.expected)
			}
		})
	}
}

func TestSwipeLeft(t *testing.T) {
	board := NewBoardFromCells([4][4]int{
		{2, 2, 0, 0},
		{0, 0, 0, 0},
		{0, 0, 0, 0},
		{0, 0, 0, 0},
	})

	results := board.Swipe(Left)

	// 2+2=4になり、空きマス15個 × 2(2か4) = 30通り
	if len(results) != 30 {
		t.Errorf("expected 30 results, got %d", len(results))
	}

	// 最初の結果を確認（左上が4になっているはず）
	if results[0].Get(0, 0) != 4 {
		t.Errorf("expected top-left to be 4, got %d", results[0].Get(0, 0))
	}
}

func TestSwipeRight(t *testing.T) {
	board := NewBoardFromCells([4][4]int{
		{2, 2, 0, 0},
		{0, 0, 0, 0},
		{0, 0, 0, 0},
		{0, 0, 0, 0},
	})

	results := board.Swipe(Right)

	// 右端に4が来るはず
	swiped, _ := board.SwipeWithoutSpawn(Right)
	if swiped.Get(0, 3) != 4 {
		t.Errorf("expected top-right to be 4, got %d", swiped.Get(0, 3))
	}

	if len(results) != 30 {
		t.Errorf("expected 30 results, got %d", len(results))
	}
}

func TestSwipeUp(t *testing.T) {
	board := NewBoardFromCells([4][4]int{
		{2, 0, 0, 0},
		{2, 0, 0, 0},
		{0, 0, 0, 0},
		{0, 0, 0, 0},
	})

	results := board.Swipe(Up)

	// 上端に4が来るはず
	swiped, _ := board.SwipeWithoutSpawn(Up)
	if swiped.Get(0, 0) != 4 {
		t.Errorf("expected top-left to be 4, got %d", swiped.Get(0, 0))
	}

	if len(results) != 30 {
		t.Errorf("expected 30 results, got %d", len(results))
	}
}

func TestSwipeDown(t *testing.T) {
	board := NewBoardFromCells([4][4]int{
		{2, 0, 0, 0},
		{2, 0, 0, 0},
		{0, 0, 0, 0},
		{0, 0, 0, 0},
	})

	results := board.Swipe(Down)

	// 下端に4が来るはず
	swiped, _ := board.SwipeWithoutSpawn(Down)
	if swiped.Get(3, 0) != 4 {
		t.Errorf("expected bottom-left to be 4, got %d", swiped.Get(3, 0))
	}

	if len(results) != 30 {
		t.Errorf("expected 30 results, got %d", len(results))
	}
}

func TestSwipeNoChange(t *testing.T) {
	board := NewBoardFromCells([4][4]int{
		{2, 0, 0, 0},
		{4, 0, 0, 0},
		{8, 0, 0, 0},
		{16, 0, 0, 0},
	})

	results := board.Swipe(Left)

	// 左にスワイプしても変化なし
	if results != nil {
		t.Errorf("expected nil results for no-change swipe, got %d results", len(results))
	}
}

func TestBoardImmutability(t *testing.T) {
	original := NewBoardFromCells([4][4]int{
		{2, 2, 0, 0},
		{0, 0, 0, 0},
		{0, 0, 0, 0},
		{0, 0, 0, 0},
	})
	originalCopy := original.Copy()

	_ = original.Swipe(Left)

	// オリジナルが変更されていないことを確認
	if !original.Equal(originalCopy) {
		t.Error("original board was mutated")
	}
}

func TestIsGameOver(t *testing.T) {
	// ゲームオーバーの盤面
	gameOver := NewBoardFromCells([4][4]int{
		{2, 4, 2, 4},
		{4, 2, 4, 2},
		{2, 4, 2, 4},
		{4, 2, 4, 2},
	})

	if !gameOver.IsGameOver() {
		t.Error("expected game over")
	}

	// ゲームオーバーではない盤面
	notGameOver := NewBoardFromCells([4][4]int{
		{2, 2, 0, 0},
		{0, 0, 0, 0},
		{0, 0, 0, 0},
		{0, 0, 0, 0},
	})

	if notGameOver.IsGameOver() {
		t.Error("expected not game over")
	}
}

func TestEqual(t *testing.T) {
	board1 := NewBoardFromCells([4][4]int{
		{2, 4, 0, 0},
		{0, 0, 0, 0},
		{0, 0, 0, 0},
		{0, 0, 0, 0},
	})

	board2 := NewBoardFromCells([4][4]int{
		{2, 4, 0, 0},
		{0, 0, 0, 0},
		{0, 0, 0, 0},
		{0, 0, 0, 0},
	})

	board3 := NewBoardFromCells([4][4]int{
		{4, 2, 0, 0},
		{0, 0, 0, 0},
		{0, 0, 0, 0},
		{0, 0, 0, 0},
	})

	if !board1.Equal(board2) {
		t.Error("expected board1 and board2 to be equal")
	}

	if board1.Equal(board3) {
		t.Error("expected board1 and board3 to be different")
	}
}

func TestString(t *testing.T) {
	board := NewBoardFromCells([4][4]int{
		{2, 4, 8, 16},
		{32, 64, 128, 256},
		{512, 1024, 2048, 0},
		{0, 0, 0, 2},
	})

	str := board.String()

	// 各値が含まれていることを確認
	for _, v := range []string{"2", "4", "8", "16", "32", "64", "128", "256", "512", "1024", "2048"} {
		if !strings.Contains(str, v) {
			t.Errorf("expected string to contain %s", v)
		}
	}

	// 罫線が含まれていることを確認
	if !strings.Contains(str, "+------+") {
		t.Error("expected string to contain border")
	}

	t.Logf("Board display:\n%s", str)
}
