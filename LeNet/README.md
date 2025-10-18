# LeNet-5 実装 (PyTorch)

MNIST 使った LeNet-5の簡単な実装
そもそも LeNetとは、「画像を認識するためのニューラルネットワークの構造（モデル）」のこと（脳みそみたいなもの）
LeNet は CNN の一種である。
LeNet は Encoder のみで終わるモデルと言える。

処理方法：
画像（２８*２８）　→ 畳み込み層（Conv2d）特徴量を抽出　ー　プーリング層（AvgPool）情報量圧縮　→ 全層結合（Linear）数字の分類

## Setup
conda activate hirokienv
pip install -r requirements.txt

LeNet/
├── data/                  ← MNISTなどのデータを保存（自動DLでもOK）
├── models/                ← モデル定義ファイル（LeNet本体）
│   └── lenet.py
├── train.py               ← 学習スクリプト（訓練ループ、評価など）
├── utils.py               ← 補助関数（ログ、精度計算など）
├── requirements.txt       ← 必要ライブラリ一覧
└── README.md              ← 説明（実行方法など）