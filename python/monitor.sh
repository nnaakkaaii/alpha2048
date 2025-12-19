#!/bin/bash

# 学習進捗モニタリングスクリプト

LOG_DIR="logs"
CHECKPOINT_DIR="checkpoints"

# 最新のログファイルを探す
LATEST_LOG=$(ls -t ${LOG_DIR}/training_*.log 2>/dev/null | head -1)

if [ -z "${LATEST_LOG}" ]; then
    echo "ログファイルが見つかりません"
    exit 1
fi

# プロセスID確認
if [ -f "${LOG_DIR}/train.pid" ]; then
    PID=$(cat ${LOG_DIR}/train.pid)
    if ps -p ${PID} > /dev/null 2>&1; then
        echo "学習プロセス実行中 (PID: ${PID})"
    else
        echo "学習プロセス停止済み (PID: ${PID})"
    fi
    echo ""
fi

# 最新の統計情報
echo "=== 最新の学習統計 (最後の10エピソード) ==="
grep "Episode" ${LATEST_LOG} | tail -10
echo ""

# 到達タイルの統計
echo "=== タイル到達統計 ==="
TOTAL_EPISODES=$(grep -c "Episode" ${LATEST_LOG})
echo "総エピソード数: ${TOTAL_EPISODES}"

# 平均スコアの推移（1000エピソードごと）
echo ""
echo "=== 平均スコアの推移 (1000エピソードごと) ==="
grep "Episode.*00 " ${LATEST_LOG} | grep -E "Episode\s+(1000|2000|3000|4000|5000|6000|7000|8000|9000|10000)" | awk '{print $2, "Episodes - Avg Score:", $6, "Max Tile:", $10}'

# 最良スコア
echo ""
echo "=== 最良記録 ==="
if [ -f "${CHECKPOINT_DIR}/best_model.pth" ]; then
    BEST_LINE=$(grep "best avg:" ${LATEST_LOG} | tail -1)
    if [ ! -z "${BEST_LINE}" ]; then
        echo "${BEST_LINE}"
    fi
fi

# ディスク使用量
echo ""
echo "=== ディスク使用状況 ==="
du -sh ${LOG_DIR}
du -sh ${CHECKPOINT_DIR}

# 最新のチェックポイント
echo ""
echo "=== 最新のチェックポイント ==="
ls -lht ${CHECKPOINT_DIR}/*.pth | head -3

echo ""
echo "ログ確認: tail -f ${LATEST_LOG}"