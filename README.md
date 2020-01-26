###Bandit Sample
MultiProcessingを用いた非定常バンディットタスクのシミュレーションプログラム
#### Discription
コマンドライン引数によってハイパーパラメータを操作

#### Usage

    cd bandit_sample
    python ./Bandit/simlation_mp.py [-s] [-u] [-t] [-w] [-n]
#### Example

    python ./Bandit/simlation_mp.py 1 1 sim_normal 1000 10


#### PARAMETER


Parameter | Type(Range)| Discription
:------------: |:-------------:|-------------
-s | int (0 or 1)|0:非定常に環境が変化 1:定常
-u | int (0 or 1)|0:環境がwstep目で変動 1:環境がw%の確率で変動
-t|str| タスク名を指定
-w|int| 環境変動率の指定
-n|int|シミュレーション回数