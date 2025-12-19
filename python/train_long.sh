#!/bin/bash

# 2048 N-tuple + TD Learning Long Training Script
# 32768タイル達成を目指す推奨設定

# 設定
MODEL="ntuple"              # N-tuple + TD学習（推奨）
EPISODES=500000             # 50万エピソード（32768達成には十分な学習が必要）
SAVE_INTERVAL=5000          # 5000エピソードごとに保存
LR=0.1                      # 初期学習率
LR_DECAY=0.999999           # 緩やかな学習率減衰（32768達成に重要）
LOG_DIR="logs"
CHECKPOINT_DIR="checkpoints"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/training_${MODEL}_${TIMESTAMP}.log"

# ディレクトリ作成
mkdir -p ${LOG_DIR}
mkdir -p ${CHECKPOINT_DIR}

# 仮想環境のアクティベート
source .venv/bin/activate

echo "=== 2048 N-tuple + TD Learning 長時間学習開始 ===" | tee -a ${LOG_FILE}
echo "開始時刻: $(date)" | tee -a ${LOG_FILE}
echo "モデル: ${MODEL}" | tee -a ${LOG_FILE}
echo "エピソード数: ${EPISODES}" | tee -a ${LOG_FILE}
echo "学習率: ${LR}" | tee -a ${LOG_FILE}
echo "学習率減衰: ${LR_DECAY}" | tee -a ${LOG_FILE}
echo "保存間隔: ${SAVE_INTERVAL}エピソード" | tee -a ${LOG_FILE}
echo "ログファイル: ${LOG_FILE}" | tee -a ${LOG_FILE}
echo "================================" | tee -a ${LOG_FILE}

# nohupで実行（バックグラウンド実行）
nohup python -u training/reinforcement_learning/train.py \
    --model ${MODEL} \
    --episodes ${EPISODES} \
    --save-interval ${SAVE_INTERVAL} \
    --lr ${LR} \
    --lr-decay ${LR_DECAY} \
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
echo "タイル達成率確認:" | tee -a ${LOG_FILE}
echo "  grep -E '(2048|4096|8192|16384|32768)' ${LOG_FILE} | tail -5" | tee -a ${LOG_FILE}
echo "" | tee -a ${LOG_FILE}
echo "プロセス確認:" | tee -a ${LOG_FILE}
echo "  ps -p ${PID}" | tee -a ${LOG_FILE}
echo "" | tee -a ${LOG_FILE}
echo "学習を停止する場合:" | tee -a ${LOG_FILE}
echo "  kill ${PID}" | tee -a ${LOG_FILE}
echo "" | tee -a ${LOG_FILE}
echo "================================" | tee -a ${LOG_FILE}
echo "" | tee -a ${LOG_FILE}
echo "期待される結果（十分な学習後）:" | tee -a ${LOG_FILE}
echo "  2048:  99%+" | tee -a ${LOG_FILE}
echo "  4096:  95%+" | tee -a ${LOG_FILE}
echo "  8192:  70%+" | tee -a ${LOG_FILE}
echo "  16384: 30%+" | tee -a ${LOG_FILE}
echo "  32768: 数%" | tee -a ${LOG_FILE}
echo "================================" | tee -a ${LOG_FILE}
