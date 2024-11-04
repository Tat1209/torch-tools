# -*- coding: utf-8 -*-
"""
glife.pyプログラム
ライフゲーム計算プログラム 
2次元セルオートマトンの一種である、ライフゲームのプログラム
結果をグラフ描画する
使い方　c:\>python glife.py　< (初期状態ファイル名)
初期状態ファイルには初期配置を記述する
"""
# モジュールのインポート
import sys  # readlines()に必要
import numpy as np
import matplotlib.pyplot as plt

# 定数
N = 400     # ライフゲームの世界の大きさ
MAXT = 5500  # 繰り返しの回数

#下請け関数の定義
# initworld()関数
def initworld(world):
    """初期値の読み込み"""
    chrworld = sys.stdin.readlines()
    # 内部表現への変換
    for no, line in enumerate(chrworld):
        line = line.rstrip()
        print(line)
        for i in range(len(line)):
            world[no][i] = int(line[i])
# initworld()関数の終わり

# nextt()関数
def nextt(world):
    """worldの状態の更新"""
    nextworld = [[0 for i in range(N)] for j in range(N)]  # 次世代
    # ルールの適用
    for i in range(1, N - 1):
        for j in range(1, N - 1):
            nextworld[i][j] = calcnext(world, i, j)
    # worldの更新
    for i in range(N):
        for j in range(N):
            world[i][j] = nextworld[i][j]
# nextt()関数の終わり

# calcnext()関数
def calcnext(world, i, j):
    """1セルの状態の更新"""
    no_of_one = 0  # 周囲にある状態1のセルの数
    # 状態1のセルを数える
    for x in range(i - 1, i + 2):
        for y in range(j - 1, j + 2):
            no_of_one += world[x][y]
    no_of_one -= world[i][j]  # 自分自身はカウントしない
    # 状態の更新
    if no_of_one == 3:
        return 1            # 誕生
    elif no_of_one == 2:
        return world[i][j]  # 存続
    return 0  # 上記以外
# calcnext()関数の終わり

# メイン実行部
world = [[0 for i in range(N)] for j in range(N)]
# world[][]への初期値の読み込み
initworld(world)
print("t=0")    # 初期時刻の出力

# グラフ出力
w = plt.imshow(world, interpolation="nearest")

plt.pause(0.01)

# 時間発展の計算
for t in range(1, MAXT):
    nextt(world)       # 次の時刻に更新
    print("t=", t)     # 時刻の出力
    # print(world)     # worldの状態の出力
    # グラフ出力
    w.set_data(world)  # 描画データの更新
    
    plt.pause(0.01)
plt.show()
# glife.pyの終わり
