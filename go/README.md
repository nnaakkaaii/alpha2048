# alpha2048 - Go実装

2048ゲームのソルバーのGo実装です。

## 遊び方

```bash
go run ./cmd/play
```

### 操作方法

| キー | 方向 |
|------|------|
| w    | 上   |
| s    | 下   |
| a    | 左   |
| d    | 右   |
| q    | 終了 |

### 画面例

```
=== 2048 ===
Controls: w=Up, s=Down, a=Left, d=Right, q=Quit

+------+------+------+------+
|    2 |    4 |    8 |   16 |
+------+------+------+------+
|   32 |   64 |  128 |  256 |
+------+------+------+------+
|  512 | 1024 | 2048 |      |
+------+------+------+------+
|      |      |      |    2 |
+------+------+------+------+
Score: 12345
Move:
```

## 自動プレイ（ソルバー）

```bash
go run ./cmd/autoplay
```

### オプション

| フラグ | デフォルト | 説明 |
|--------|-----------|------|
| `-depth` | 3 | 探索深さ |
| `-delay` | 100 | 手の間隔（ミリ秒） |
| `-astar` | false | A*アルゴリズムを使用 |
| `-bitboard` | false | BitBoard表現を使用（高速） |
| `-quiet` | false | 出力を抑制 |

### 例

```bash
# 深さ4で探索（より賢いが遅い）
go run ./cmd/autoplay -depth=4

# BitBoardを使用（約8倍高速）
go run ./cmd/autoplay -bitboard -depth=5

# A*アルゴリズムを使用
go run ./cmd/autoplay -astar

# 高速実行（出力なし）
go run ./cmd/autoplay -delay=0 -quiet
```

## 盤面分析ツール

インタラクティブに盤面を分析し、最適な手を提案します。

```bash
go run ./cmd/analyze
```

### オプション

| フラグ | デフォルト | 説明 |
|--------|-----------|------|
| `-depth` | 5 | 初期探索深さ |
| `-bitboard` | false | BitBoard表現を使用 |
| `-astar` | false | A*アルゴリズムを使用 |

## 開発

### ビルド

```bash
# 各コマンドのビルド
go build ./cmd/play
go build ./cmd/autoplay
go build ./cmd/analyze
```

### テスト

```bash
go test ./...
```

### パフォーマンス

- 通常のBoard表現: 基本実装
- BitBoard表現: 約8倍の高速化
- A*探索: ヒューリスティクスによる枝刈り

## アーキテクチャ

- `cmd/` - 各種コマンドのエントリーポイント
- `internal/domain/` - ゲームロジック、評価関数、ソルバー
- `internal/usecase/` - CLIとの対話、表示制御