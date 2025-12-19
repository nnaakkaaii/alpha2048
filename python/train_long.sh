#!/bin/bash

# 2048 DQN Long Training Script (1日放置用)
# 1024タイルを安定して生成できるモデルを目指す

# 設定
EPISODES=50000  # 50000エピソード（約1日想定）
SAVE_INTERVAL=1000  # 1000エピソードごとに保存
LOG_DIR="logs"
CHECKPOINT_DIR="checkpoints"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/training_${TIMESTAMP}.log"

# ディレクトリ作成
mkdir -p ${LOG_DIR}
mkdir -p ${CHECKPOINT_DIR}

# 仮想環境のアクティベート
source .venv/bin/activate

# GPU/CPU自動選択
echo "=== 2048 DQN長時間学習開始 ===" | tee -a ${LOG_FILE}
echo "開始時刻: $(date)" | tee -a ${LOG_FILE}
echo "エピソード数: ${EPISODES}" | tee -a ${LOG_FILE}
echo "保存間隔: ${SAVE_INTERVAL}エピソード" | tee -a ${LOG_FILE}
echo "ログファイル: ${LOG_FILE}" | tee -a ${LOG_FILE}
echo "================================" | tee -a ${LOG_FILE}

# nohupで実行（バックグラウンド実行）
nohup python -u training/reinforcement_learning/train.py \
    --episodes ${EPISODES} \
    --save-interval ${SAVE_INTERVAL} \
    --device auto \
    >> ${LOG_FILE} 2>&1 &

# プロセスIDを保存
PID=$!
echo ${PID} > ${LOG_DIR}/train.pid
echo "学習プロセスID: ${PID}" | tee -a ${LOG_FILE}

echo "" | tee -a ${LOG_FILE}
echo "=== 学習がバックグラウンドで開始されました ===" | tee -a ${LOG_FILE}
echo "" | tee -a ${LOG_FILE}
echo "進捗確認方法:" | tee -a ${LOG_FILE}
echo "  tail -f ${LOG_FILE}" | tee -a ${LOG_FILE}
echo "" | tee -a ${LOG_FILE}
echo "最新の統計確認:" | tee -a ${LOG_FILE}
echo "  grep 'Episode' ${LOG_FILE} | tail -10" | tee -a ${LOG_FILE}
echo "" | tee -a ${LOG_FILE}
echo "プロセス確認:" | tee -a ${LOG_FILE}
echo "  ps -p ${PID}" | tee -a ${LOG_FILE}
echo "" | tee -a ${LOG_FILE}
echo "学習を停止する場合:" | tee -a ${LOG_FILE}
echo "  kill ${PID}" | tee -a ${LOG_FILE}
echo "" | tee -a ${LOG_FILE}
echo "モデルテスト（別ターミナルで実行可能）:" | tee -a ${LOG_FILE}
echo "  source .venv/bin/activate" | tee -a ${LOG_FILE}
echo "  python training/reinforcement_learning/test.py --games 100" | tee -a ${LOG_FILE}
echo "" | tee -a ${LOG_FILE}
echo "最良モデルのテスト:" | tee -a ${LOG_FILE}
echo "  python training/reinforcement_learning/test.py --model checkpoints/best_model.pth --games 100" | tee -a ${LOG_FILE}
echo "================================" | tee -a ${LOG_FILE}