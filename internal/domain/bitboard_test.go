package domain

import (
	"testing"
)

func TestBitBoardConversion(t *testing.T) {
	// テスト用の盤面を作成
	board := NewBoardFromCells([4][4]int{
		{2, 4, 8, 16},
		{32, 64, 128, 256},
		{512, 1024, 2048, 4096},
		{0, 0, 0, 0},
	})
	
	// BitBoardに変換して戻す
	bb := NewBitBoard(board)
	restored := bb.ToBoard()
	
	// 元の盤面と一致するか確認
	for r := 0; r < 4; r++ {
		for c := 0; c < 4; c++ {
			if board.Get(r, c) != restored.Get(r, c) {
				t.Errorf("Mismatch at (%d,%d): expected %d, got %d", 
					r, c, board.Get(r, c), restored.Get(r, c))
			}
		}
	}
}

func TestBitBoardSwipe(t *testing.T) {
	board := NewBoardFromCells([4][4]int{
		{2, 2, 4, 8},
		{4, 0, 4, 0},
		{8, 8, 0, 0},
		{0, 0, 0, 0},
	})
	
	bb := NewBitBoard(board)
	
	// 左スワイプのテスト
	result, score := bb.SwipeLeft()
	resultBoard := result.ToBoard()
	
	expected := NewBoardFromCells([4][4]int{
		{4, 4, 8, 0},
		{8, 0, 0, 0},
		{16, 0, 0, 0},
		{0, 0, 0, 0},
	})
	
	for r := 0; r < 4; r++ {
		for c := 0; c < 4; c++ {
			if expected.Get(r, c) != resultBoard.Get(r, c) {
				t.Errorf("SwipeLeft mismatch at (%d,%d): expected %d, got %d",
					r, c, expected.Get(r, c), resultBoard.Get(r, c))
			}
		}
	}
	
	expectedScore := 4 + 8 + 16
	if score != expectedScore {
		t.Errorf("Score mismatch: expected %d, got %d", expectedScore, score)
	}
}

func BenchmarkBitBoardSwipe(b *testing.B) {
	board := NewBoardFromCells([4][4]int{
		{2, 4, 8, 16},
		{32, 64, 128, 256},
		{512, 1024, 2048, 4096},
		{8192, 16384, 32768, 0},
	})
	bb := NewBitBoard(board)
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		bb.SwipeLeft()
		bb.SwipeRight()
		bb.SwipeUp()
		bb.SwipeDown()
	}
}

func BenchmarkNormalBoardSwipe(b *testing.B) {
	board := NewBoardFromCells([4][4]int{
		{2, 4, 8, 16},
		{32, 64, 128, 256},
		{512, 1024, 2048, 4096},
		{8192, 16384, 32768, 0},
	})
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		board.SwipeWithoutSpawn(Left)
		board.SwipeWithoutSpawn(Right)
		board.SwipeWithoutSpawn(Up)
		board.SwipeWithoutSpawn(Down)
	}
}