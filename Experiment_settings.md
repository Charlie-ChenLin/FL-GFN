# 需要考虑的实验
- set generation， 固定轨迹长度
- bag generation， 固定轨迹长度, generate a bag with a maximum capacity of 15
- grid domain， 不是很确定，但是这个是轨迹作为一个DAG，可以借鉴到MOE里头
- molecule generation， mode_thres=7.5(or 8), maximum trajectory length 8, num of actions varies
- RNA sequence generation, maximum length 8, num of actions 4
- maximum independent set problem
- bit sequence generation， 这个好像是length不一样的？看是increasing length， 好像每个子任务内trajectory length还是一样的


# 可能的实验汇总

## 固定长度的实验
- set generation(3 size settings: small, medium, large)
- bag generation
- bit sequence generation(autoregressive,TB paper, Tiapkin et al 2024)(length=120) 

## 尚不明确
- molecule generation(=small drug molecule synthesis,Tiapkin et al 2024)
- maximum independent set problem 这个实验长度没有意义
- Hypergrid(TB和最早的文章,Tiapkin et al 2024)
- RNA sequence generation
- AMP generation(autoregressive, TB paper)(max length=60, vocab size=20)

## 不固定长度
- text generation

# Set Generation
## Env
python=3.12
其余要求均为最新
torch装的cuda 12.4

## Exp setting
seed=1
默认的case：alpha=0.5，unbiased
mode的threshold是多少啊？

  