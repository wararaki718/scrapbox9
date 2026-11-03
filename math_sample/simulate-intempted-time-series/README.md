# Interrupted Time Series Simulation

介入の前後で時系列の水準と傾きがどう変化したかを、分割回帰で確認するサンプルです。月次24件の合成データをコード内で決定論的に作成し、13か月目を介入時点とします。

## 分析モデル

次のOLSモデルを推定します。

```text
Y_t = beta_0 + beta_1 * t + beta_2 * post_t + beta_3 * time_after_t + error_t
```

- `beta_0`: 介入前の切片
- `beta_1`: 介入前の傾き
- `beta_2`: 介入時点での水準変化
- `beta_3`: 介入後の傾きの変化
- 介入後の傾き: `beta_1 + beta_3`

`post_t` は介入時点以降なら1、それ以外なら0です。`time_after_t` は介入後の経過月数（介入前は0）です。

実行すると、係数、介入による水準・傾き変化のp値と95%信頼区間を表示し、観測値・推定値・介入時点を `interrupted_time_series.png` に保存します。

## 実行方法

```shell
python -m pip install -r requirements.txt
python main.py
```

## 注意点

これは手法を説明するための合成データ例です。単純なOLSのため、自己相関、季節性、交絡要因は調整していません。係数だけで介入の因果効果が確定するわけではありません。
