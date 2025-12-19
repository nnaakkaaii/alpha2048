package domain

import (
	"testing"
)

func TestEmptyCellsEvaluator(t *testing.T) {
	ev := &EmptyCellsEvaluator{}

	board := NewBoardFromCells([4][4]int{
		{2, 0, 0, 0},
		{0, 0, 0, 0},
		{0, 0, 0, 0},
		{0, 0, 0, 0},
	})

	score := ev.Evaluate(board)
	if score != 15 {
		t.Errorf("expected 15 empty cells, got %f", score)
	}
}

func TestCornerBonusEvaluator(t *testing.T) {
	ev := &CornerBonusEvaluator{}

	// 角に最大値
	cornerBoard := NewBoardFromCells([4][4]int{
		{16, 2, 0, 0},
		{0, 0, 0, 0},
		{0, 0, 0, 0},
		{0, 0, 0, 0},
	})
	if ev.Evaluate(cornerBoard) != 1.0 {
		t.Error("expected corner bonus 1.0")
	}

	// 角以外に最大値
	nonCornerBoard := NewBoardFromCells([4][4]int{
		{2, 16, 0, 0},
		{0, 0, 0, 0},
		{0, 0, 0, 0},
		{0, 0, 0, 0},
	})
	if ev.Evaluate(nonCornerBoard) != 0.0 {
		t.Error("expected corner bonus 0.0")
	}
}

func TestSmoothnessEvaluator(t *testing.T) {
	ev := &SmoothnessEvaluator{}

	// 滑らかな盤面（隣接が近い値）
	smoothBoard := NewBoardFromCells([4][4]int{
		{2, 4, 0, 0},
		{0, 0, 0, 0},
		{0, 0, 0, 0},
		{0, 0, 0, 0},
	})

	// 荒い盤面（隣接が遠い値）
	roughBoard := NewBoardFromCells([4][4]int{
		{2, 1024, 0, 0},
		{0, 0, 0, 0},
		{0, 0, 0, 0},
		{0, 0, 0, 0},
	})

	smoothScore := ev.Evaluate(smoothBoard)
	roughScore := ev.Evaluate(roughBoard)

	// 滑らかな方がスコアが高い（ペナルティが小さい）
	if smoothScore <= roughScore {
		t.Errorf("smooth board should have higher score: smooth=%f, rough=%f", smoothScore, roughScore)
	}
}

func TestMonotonicityEvaluator(t *testing.T) {
	ev := &MonotonicityEvaluator{}

	// 単調な盤面
	monoBoard := NewBoardFromCells([4][4]int{
		{16, 8, 4, 2},
		{0, 0, 0, 0},
		{0, 0, 0, 0},
		{0, 0, 0, 0},
	})

	// 非単調な盤面
	nonMonoBoard := NewBoardFromCells([4][4]int{
		{2, 16, 4, 8},
		{0, 0, 0, 0},
		{0, 0, 0, 0},
		{0, 0, 0, 0},
	})

	monoScore := ev.Evaluate(monoBoard)
	nonMonoScore := ev.Evaluate(nonMonoBoard)

	if monoScore <= nonMonoScore {
		t.Errorf("monotonic board should have higher score: mono=%f, nonMono=%f", monoScore, nonMonoScore)
	}
}

func TestSnakePatternEvaluator(t *testing.T) {
	ev := &SnakePatternEvaluator{}

	// スネークパターンに近い盤面
	snakeBoard := NewBoardFromCells([4][4]int{
		{2048, 1024, 512, 256},
		{16, 32, 64, 128},
		{8, 4, 2, 0},
		{0, 0, 0, 0},
	})

	score := ev.Evaluate(snakeBoard)
	if score <= 0 {
		t.Errorf("snake pattern board should have positive score, got %f", score)
	}
}

func TestWeightedEvaluator(t *testing.T) {
	evaluators := []Evaluator{
		&EmptyCellsEvaluator{},
		&CornerBonusEvaluator{},
	}
	weights := []float64{1.0, 10.0}

	wev := NewWeightedEvaluator(evaluators, weights)

	board := NewBoardFromCells([4][4]int{
		{16, 0, 0, 0},
		{0, 0, 0, 0},
		{0, 0, 0, 0},
		{0, 0, 0, 0},
	})

	// 15 empty cells * 1.0 + corner bonus 1.0 * 10.0 = 25.0
	score := wev.Evaluate(board)
	expected := 15.0*1.0 + 1.0*10.0
	if score != expected {
		t.Errorf("expected %f, got %f", expected, score)
	}
}
