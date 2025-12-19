#!/bin/bash

# 最良モデルの性能テストスクリプト
# 1024タイル到達率を詳細に分析

source .venv/bin/activate

GAMES=100
MODEL_PATH="checkpoints/best_model.pth"
DEFAULT_MODEL="checkpoints/model.pth"

# モデル選択
if [ -f "${MODEL_PATH}" ]; then
    echo "最良モデルをテスト: ${MODEL_PATH}"
    MODEL=${MODEL_PATH}
elif [ -f "${DEFAULT_MODEL}" ]; then
    echo "最良モデルが見つかりません。通常モデルをテスト: ${DEFAULT_MODEL}"
    MODEL=${DEFAULT_MODEL}
else
    echo "モデルファイルが見つかりません"
    exit 1
fi

echo "テストゲーム数: ${GAMES}"
echo "================================"
echo ""

# テスト実行と結果保存
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULT_FILE="logs/test_results_${TIMESTAMP}.txt"
mkdir -p logs

python training/reinforcement_learning/test.py \
    --model ${MODEL} \
    --games ${GAMES} \
    --device auto \
    | tee ${RESULT_FILE}

# 結果の詳細分析
echo ""
echo "=== 詳細分析 ==="

# 1024以上の達成率
REACHED_1024=$(grep "Reached 1024:" ${RESULT_FILE} | tail -1)
REACHED_512=$(grep "Reached 512:" ${RESULT_FILE} | tail -1)

echo "${REACHED_1024}"
echo "${REACHED_512}"

# スコア分布
echo ""
echo "=== スコア統計 ==="
grep -E "Avg Score:|Max Score:|Min Score:" ${RESULT_FILE}

echo ""
echo "詳細結果: ${RESULT_FILE}"