package domain

import "math"

// Evaluator はBoardを評価してスコアを返すインターフェース
type Evaluator interface {
	Evaluate(b Board) float64
}

// WeightedEvaluator は複数のEvaluatorを係数付きで組み合わせる
type WeightedEvaluator struct {
	evaluators []Evaluator
	weights    []float64
}

// NewWeightedEvaluator は係数付きEvaluatorを生成する
func NewWeightedEvaluator(evaluators []Evaluator, weights []float64) *WeightedEvaluator {
	return &WeightedEvaluator{
		evaluators: evaluators,
		weights:    weights,
	}
}

// Evaluate は全てのEvaluatorの重み付き和を返す
func (w *WeightedEvaluator) Evaluate(b Board) float64 {
	score := 0.0
	for i, ev := range w.evaluators {
		score += w.weights[i] * ev.Evaluate(b)
	}
	return score
}

// EmptyCellsEvaluator は空きマス数で評価する
type EmptyCellsEvaluator struct{}

func (e *EmptyCellsEvaluator) Evaluate(b Board) float64 {
	return float64(len(b.EmptyCells()))
}

// MonotonicityEvaluator は単調性で評価する（角から降順に並ぶほど高評価）
type MonotonicityEvaluator struct{}

func (e *MonotonicityEvaluator) Evaluate(b Board) float64 {
	// 4つの角それぞれを基準にした単調性を計算し、最大を返す
	scores := []float64{
		e.calcMonotonicity(b, true, true),   // 左上基準
		e.calcMonotonicity(b, true, false),  // 右上基準
		e.calcMonotonicity(b, false, true),  // 左下基準
		e.calcMonotonicity(b, false, false), // 右下基準
	}
	maxScore := scores[0]
	for _, s := range scores[1:] {
		if s > maxScore {
			maxScore = s
		}
	}
	return maxScore
}

func (e *MonotonicityEvaluator) calcMonotonicity(b Board, fromTop, fromLeft bool) float64 {
	score := 0.0

	// 行方向の単調性
	for r := 0; r < 4; r++ {
		for c := 0; c < 3; c++ {
			c1, c2 := c, c+1
			if !fromLeft {
				c1, c2 = 3-c, 3-c-1
			}
			v1, v2 := b.Get(r, c1), b.Get(r, c2)
			if v1 >= v2 {
				score += 1
			}
		}
	}

	// 列方向の単調性
	for c := 0; c < 4; c++ {
		for r := 0; r < 3; r++ {
			r1, r2 := r, r+1
			if !fromTop {
				r1, r2 = 3-r, 3-r-1
			}
			v1, v2 := b.Get(r1, c), b.Get(r2, c)
			if v1 >= v2 {
				score += 1
			}
		}
	}

	return score
}

// SmoothnessEvaluator は隣接タイルの値の差で評価する（差が小さいほど高評価）
type SmoothnessEvaluator struct{}

func (e *SmoothnessEvaluator) Evaluate(b Board) float64 {
	penalty := 0.0

	for r := 0; r < 4; r++ {
		for c := 0; c < 4; c++ {
			v := b.Get(r, c)
			if v == 0 {
				continue
			}
			logV := math.Log2(float64(v))

			// 右隣
			if c < 3 {
				right := b.Get(r, c+1)
				if right != 0 {
					penalty += math.Abs(logV - math.Log2(float64(right)))
				}
			}
			// 下隣
			if r < 3 {
				down := b.Get(r+1, c)
				if down != 0 {
					penalty += math.Abs(logV - math.Log2(float64(down)))
				}
			}
		}
	}

	// ペナルティなので負の値を返す（小さいほど良い → 大きいスコア）
	return -penalty
}

// CornerBonusEvaluator は最大タイルが角にあると高評価
type CornerBonusEvaluator struct{}

func (e *CornerBonusEvaluator) Evaluate(b Board) float64 {
	maxVal := 0
	maxRow, maxCol := 0, 0

	for r := 0; r < 4; r++ {
		for c := 0; c < 4; c++ {
			v := b.Get(r, c)
			if v > maxVal {
				maxVal = v
				maxRow, maxCol = r, c
			}
		}
	}

	// 角にあれば1.0、そうでなければ0.0
	isCorner := (maxRow == 0 || maxRow == 3) && (maxCol == 0 || maxCol == 3)
	if isCorner {
		return 1.0
	}
	return 0.0
}

// SnakePatternEvaluator はスネークパターンに沿った配置を高評価
type SnakePatternEvaluator struct{}

// スネークパターンの重み（左上から蛇状に降順）
var snakeWeights = [4][4]float64{
	{15, 14, 13, 12},
	{8, 9, 10, 11},
	{7, 6, 5, 4},
	{0, 1, 2, 3},
}

func (e *SnakePatternEvaluator) Evaluate(b Board) float64 {
	// 4つの回転・反転パターンを試して最大を返す
	patterns := [][4][4]float64{
		snakeWeights,
		rotateWeights(snakeWeights),
		rotateWeights(rotateWeights(snakeWeights)),
		rotateWeights(rotateWeights(rotateWeights(snakeWeights))),
	}

	// 水平反転も追加
	for i := 0; i < 4; i++ {
		patterns = append(patterns, flipHorizontal(patterns[i]))
	}

	maxScore := math.Inf(-1)
	for _, pattern := range patterns {
		score := e.calcPatternScore(b, pattern)
		if score > maxScore {
			maxScore = score
		}
	}
	return maxScore
}

func (e *SnakePatternEvaluator) calcPatternScore(b Board, weights [4][4]float64) float64 {
	score := 0.0
	for r := 0; r < 4; r++ {
		for c := 0; c < 4; c++ {
			v := b.Get(r, c)
			if v > 0 {
				score += weights[r][c] * math.Log2(float64(v))
			}
		}
	}
	return score
}

func rotateWeights(w [4][4]float64) [4][4]float64 {
	var result [4][4]float64
	for r := 0; r < 4; r++ {
		for c := 0; c < 4; c++ {
			result[c][3-r] = w[r][c]
		}
	}
	return result
}

func flipHorizontal(w [4][4]float64) [4][4]float64 {
	var result [4][4]float64
	for r := 0; r < 4; r++ {
		for c := 0; c < 4; c++ {
			result[r][3-c] = w[r][c]
		}
	}
	return result
}

// MaxTileEvaluator は最大タイルの値（log2）で評価する
type MaxTileEvaluator struct{}

func (e *MaxTileEvaluator) Evaluate(b Board) float64 {
	maxVal := 0
	for r := 0; r < 4; r++ {
		for c := 0; c < 4; c++ {
			v := b.Get(r, c)
			if v > maxVal {
				maxVal = v
			}
		}
	}
	if maxVal == 0 {
		return 0
	}
	return math.Log2(float64(maxVal))
}

// LargestTilePotentialEvaluator は最大タイルを作る可能性で評価する
// 単一の最大タイルの価値を最大化することに特化
type LargestTilePotentialEvaluator struct{}

func (e *LargestTilePotentialEvaluator) Evaluate(b Board) float64 {
	score := 0.0

	// 最大タイルとその位置を特定
	maxVal := 0
	maxRow, maxCol := 0, 0
	for r := 0; r < 4; r++ {
		for c := 0; c < 4; c++ {
			v := b.Get(r, c)
			if v > maxVal {
				maxVal = v
				maxRow, maxCol = r, c
			}
		}
	}

	// 最大タイルの価値（指数的に重要）
	if maxVal > 0 {
		score += float64(maxVal) * 10.0
	}

	// 最大タイルが角にあるとボーナス（大きいタイルほどボーナスも大きい）
	isCorner := (maxRow == 0 || maxRow == 3) && (maxCol == 0 || maxCol == 3)
	if isCorner && maxVal > 0 {
		score += float64(maxVal) * 5.0
	}

	// 空きマス数（機動力確保）
	emptyCells := b.EmptyCells()
	score += float64(len(emptyCells)) * float64(maxVal) * 0.1

	// 最大タイルの周囲に次に大きいタイルがあるとボーナス（マージの可能性）
	secondMax := 0
	for r := 0; r < 4; r++ {
		for c := 0; c < 4; c++ {
			v := b.Get(r, c)
			if v > secondMax && v < maxVal {
				secondMax = v
			}
		}
	}

	// 最大タイルと同じ値が隣接していればマージ可能で大ボーナス
	neighbors := [][2]int{{-1, 0}, {1, 0}, {0, -1}, {0, 1}}
	for _, d := range neighbors {
		nr, nc := maxRow+d[0], maxCol+d[1]
		if nr >= 0 && nr < 4 && nc >= 0 && nc < 4 {
			v := b.Get(nr, nc)
			if v == maxVal {
				// 最大タイル同士が隣接 = 次のレベルへ進める
				score += float64(maxVal) * 20.0
			} else if v == secondMax && secondMax > 0 {
				// 2番目に大きいタイルが隣接
				score += float64(secondMax) * 2.0
			}
		}
	}

	// 単調性ボーナス（角から降順に並んでいると良い）
	score += e.cornerMonotonicity(b, maxRow, maxCol) * float64(maxVal) * 0.5

	return score
}

// cornerMonotonicity は最大タイルの角を基準にした単調性を計算
func (e *LargestTilePotentialEvaluator) cornerMonotonicity(b Board, maxRow, maxCol int) float64 {
	// 最大タイルがある角を特定（なければ最も近い角）
	cornerRow := 0
	if maxRow > 1 {
		cornerRow = 3
	}
	cornerCol := 0
	if maxCol > 1 {
		cornerCol = 3
	}

	mono := 0.0
	// 行方向
	for r := 0; r < 4; r++ {
		for c := 0; c < 3; c++ {
			c1, c2 := c, c+1
			if cornerCol == 3 {
				c1, c2 = 3-c, 3-c-1
			}
			v1, v2 := b.Get(r, c1), b.Get(r, c2)
			if v1 >= v2 {
				mono += 1
			}
		}
	}
	// 列方向
	for c := 0; c < 4; c++ {
		for r := 0; r < 3; r++ {
			r1, r2 := r, r+1
			if cornerRow == 3 {
				r1, r2 = 3-r, 3-r-1
			}
			v1, v2 := b.Get(r1, c), b.Get(r2, c)
			if v1 >= v2 {
				mono += 1
			}
		}
	}
	return mono / 24.0 // 正規化（最大24）
}

// MergeableEvaluator は隣接する同じ値のペア数で評価する
type MergeableEvaluator struct{}

func (e *MergeableEvaluator) Evaluate(b Board) float64 {
	count := 0.0
	for r := 0; r < 4; r++ {
		for c := 0; c < 4; c++ {
			v := b.Get(r, c)
			if v == 0 {
				continue
			}
			// 右隣
			if c < 3 && b.Get(r, c+1) == v {
				count++
			}
			// 下隣
			if r < 3 && b.Get(r+1, c) == v {
				count++
			}
		}
	}
	return count
}
