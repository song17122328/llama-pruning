# 微调结果汇总

总计: 18 个模型配置

## 主要指标概览

| 模型               | 配置       |   参数量(B) | 微调前PPL   |   微调后PPL | PPL变化(%)   | 微调前平均ACC   |   微调后平均ACC | ACC变化(%)   |
|:-----------------|:---------|---------:|:---------|---------:|:-----------|:-----------|-----------:|:-----------|
| Llama            | base     |     8.03 | N/A      |     6.1  | N/A        | N/A        |     0.6997 | N/A        |
| Llama            | best_acc |     6.42 | 13.39    |    10.78 | -19.46     | 0.5946     |     0.6492 | 9.19       |
| Llama            | best_ppl |     6.42 | 11.82    |    10.18 | -13.91     | 0.5665     |     0.6311 | 11.41      |
| Llama-Instruct   | base     |     8.03 | N/A      |     8.26 | N/A        | N/A        |     0.699  | N/A        |
| Llama-Instruct   | best_acc |     6.42 | 13.29    |    11.28 | -15.1      | 0.6318     |     0.6477 | 2.52       |
| Llama-Instruct   | best_ppl |     6.42 | 13.2     |    11.28 | -14.57     | 0.6247     |     0.6487 | 3.84       |
| Mistral          | base     |     7.25 | N/A      |     5.38 | N/A        | N/A        |     0.7047 | N/A        |
| Mistral          | best_acc |     5.8  | 15.22    |    10.46 | -31.27     | 0.5861     |     0.6246 | 6.57       |
| Mistral          | best_ppl |     5.8  | 9.69     |     7.68 | -20.78     | 0.5332     |     0.6105 | 14.5       |
| Mistral-Instruct | base     |     7.25 | N/A      |     5.56 | N/A        | N/A        |     0.7348 | N/A        |
| Mistral-Instruct | best_acc |     5.8  | 24.33    |    12.15 | -50.05     | 0.6552     |     0.6653 | 1.55       |
| Mistral-Instruct | best_ppl |     5.8  | 9.09     |     7.79 | -14.34     | 0.6023     |     0.6375 | 5.85       |
| Qwen             | base     |     7.62 | N/A      |     6.81 | N/A        | N/A        |     0.7022 | N/A        |
| Qwen             | best_acc |     6.09 | 14.46    |    11.4  | -21.15     | 0.6094     |     0.6476 | 6.27       |
| Qwen             | best_ppl |     6.09 | 11.45    |    10.32 | -9.87      | 0.5633     |     0.6197 | 10.01      |
| Qwen-Instruct    | base     |     7.62 | N/A      |     7.39 | N/A        | N/A        |     0.7174 | N/A        |
| Qwen-Instruct    | best_acc |     6.09 | 13.42    |    10.62 | -20.86     | 0.6202     |     0.6504 | 4.87       |
| Qwen-Instruct    | best_ppl |     6.09 | 12.69    |    10.45 | -17.67     | 0.6116     |     0.6354 | 3.9        |

## 各任务详细准确率

### BOOLQ

| 模型               | 配置       | 微调前    |    微调后 | 变化      |
|:-----------------|:---------|:-------|-------:|:--------|
| Llama            | base     | N/A    | 0.8131 | N/A     |
| Llama            | best_acc | 0.6642 | 0.8174 | 0.1532  |
| Llama            | best_ppl | 0.6306 | 0.7599 | 0.1293  |
| Llama-Instruct   | base     | N/A    | 0.8306 | N/A     |
| Llama-Instruct   | best_acc | 0.8309 | 0.833  | 0.0021  |
| Llama-Instruct   | best_ppl | 0.8211 | 0.8229 | 0.0018  |
| Mistral          | base     | N/A    | 0.8214 | N/A     |
| Mistral          | best_acc | 0.7642 | 0.7994 | 0.0352  |
| Mistral          | best_ppl | 0.6505 | 0.7021 | 0.0516  |
| Mistral-Instruct | base     | N/A    | 0.8587 | N/A     |
| Mistral-Instruct | best_acc | 0.7462 | 0.8083 | 0.0621  |
| Mistral-Instruct | best_ppl | 0.6361 | 0.7508 | 0.1147  |
| Qwen             | base     | N/A    | 0.8465 | N/A     |
| Qwen             | best_acc | 0.8147 | 0.8046 | -0.0101 |
| Qwen             | best_ppl | 0.5654 | 0.6823 | 0.1169  |
| Qwen-Instruct    | base     | N/A    | 0.8642 | N/A     |
| Qwen-Instruct    | best_acc | 0.77   | 0.8055 | 0.0355  |
| Qwen-Instruct    | best_ppl | 0.6878 | 0.7737 | 0.0859  |

### PIQA

| 模型               | 配置       | 微调前    |    微调后 | 变化      |
|:-----------------|:---------|:-------|-------:|:--------|
| Llama            | base     | N/A    | 0.8079 | N/A     |
| Llama            | best_acc | 0.7437 | 0.7666 | 0.0229  |
| Llama            | best_ppl | 0.7361 | 0.7682 | 0.0321  |
| Llama-Instruct   | base     | N/A    | 0.7862 | N/A     |
| Llama-Instruct   | best_acc | 0.7399 | 0.7486 | 0.0087  |
| Llama-Instruct   | best_ppl | 0.7399 | 0.7535 | 0.0136  |
| Mistral          | base     | N/A    | 0.8226 | N/A     |
| Mistral          | best_acc | 0.7182 | 0.7361 | 0.0179  |
| Mistral          | best_ppl | 0.7612 | 0.7905 | 0.0293  |
| Mistral-Instruct | base     | N/A    | 0.8264 | N/A     |
| Mistral-Instruct | best_acc | 0.8041 | 0.8025 | -0.0016 |
| Mistral-Instruct | best_ppl | 0.7709 | 0.7927 | 0.0218  |
| Qwen             | base     | N/A    | 0.7982 | N/A     |
| Qwen             | best_acc | 0.7285 | 0.7535 | 0.025   |
| Qwen             | best_ppl | 0.7476 | 0.7666 | 0.019   |
| Qwen-Instruct    | base     | N/A    | 0.8036 | N/A     |
| Qwen-Instruct    | best_acc | 0.7459 | 0.7622 | 0.0163  |
| Qwen-Instruct    | best_ppl | 0.7677 | 0.7715 | 0.0038  |

### HELLASWAG

| 模型               | 配置       | 微调前    |    微调后 | 变化     |
|:-----------------|:---------|:-------|-------:|:-------|
| Llama            | base     | N/A    | 0.7916 | N/A    |
| Llama            | best_acc | 0.6391 | 0.701  | 0.0619 |
| Llama            | best_ppl | 0.6226 | 0.6926 | 0.07   |
| Llama-Instruct   | base     | N/A    | 0.7581 | N/A    |
| Llama-Instruct   | best_acc | 0.6562 | 0.7002 | 0.044  |
| Llama-Instruct   | best_ppl | 0.6466 | 0.6994 | 0.0528 |
| Mistral          | base     | N/A    | 0.8042 | N/A    |
| Mistral          | best_acc | 0.6231 | 0.6958 | 0.0727 |
| Mistral          | best_ppl | 0.5777 | 0.7093 | 0.1316 |
| Mistral-Instruct | base     | N/A    | 0.8293 | N/A    |
| Mistral-Instruct | best_acc | 0.7297 | 0.7552 | 0.0255 |
| Mistral-Instruct | best_ppl | 0.6752 | 0.7255 | 0.0503 |
| Qwen             | base     | N/A    | 0.7892 | N/A    |
| Qwen             | best_acc | 0.6405 | 0.6957 | 0.0552 |
| Qwen             | best_ppl | 0.6548 | 0.7023 | 0.0475 |
| Qwen-Instruct    | base     | N/A    | 0.8046 | N/A    |
| Qwen-Instruct    | best_acc | 0.6851 | 0.7113 | 0.0262 |
| Qwen-Instruct    | best_ppl | 0.6886 | 0.7138 | 0.0252 |

### WINOGRANDE

| 模型               | 配置       | 微调前    |    微调后 | 变化      |
|:-----------------|:---------|:-------|-------:|:--------|
| Llama            | base     | N/A    | 0.7261 | N/A     |
| Llama            | best_acc | 0.6906 | 0.6906 | 0.0     |
| Llama            | best_ppl | 0.6709 | 0.6803 | 0.0094  |
| Llama-Instruct   | base     | N/A    | 0.7206 | N/A     |
| Llama-Instruct   | best_acc | 0.693  | 0.6922 | -0.0008 |
| Llama-Instruct   | best_ppl | 0.6843 | 0.6946 | 0.0103  |
| Mistral          | base     | N/A    | 0.7372 | N/A     |
| Mistral          | best_acc | 0.6401 | 0.6622 | 0.0221  |
| Mistral          | best_ppl | 0.588  | 0.6164 | 0.0284  |
| Mistral-Instruct | base     | N/A    | 0.7419 | N/A     |
| Mistral-Instruct | best_acc | 0.6077 | 0.6622 | 0.0545  |
| Mistral-Instruct | best_ppl | 0.603  | 0.614  | 0.011   |
| Qwen             | base     | N/A    | 0.7285 | N/A     |
| Qwen             | best_acc | 0.6709 | 0.6748 | 0.0039  |
| Qwen             | best_ppl | 0.6251 | 0.6377 | 0.0126  |
| Qwen-Instruct    | base     | N/A    | 0.7064 | N/A     |
| Qwen-Instruct    | best_acc | 0.6322 | 0.6693 | 0.0371  |
| Qwen-Instruct    | best_ppl | 0.5943 | 0.6101 | 0.0158  |

### ARC_EASY

| 模型               | 配置       | 微调前    |    微调后 | 变化      |
|:-----------------|:---------|:-------|-------:|:--------|
| Llama            | base     | N/A    | 0.7765 | N/A     |
| Llama            | best_acc | 0.6216 | 0.7071 | 0.0855  |
| Llama            | best_ppl | 0.5694 | 0.681  | 0.1116  |
| Llama-Instruct   | base     | N/A    | 0.798  | N/A     |
| Llama-Instruct   | best_acc | 0.6751 | 0.6923 | 0.0172  |
| Llama-Instruct   | best_ppl | 0.6679 | 0.6978 | 0.0299  |
| Mistral          | base     | N/A    | 0.7824 | N/A     |
| Mistral          | best_acc | 0.6002 | 0.6671 | 0.0669  |
| Mistral          | best_ppl | 0.4891 | 0.6591 | 0.17    |
| Mistral-Instruct | base     | N/A    | 0.8262 | N/A     |
| Mistral-Instruct | best_acc | 0.7534 | 0.7269 | -0.0265 |
| Mistral-Instruct | best_ppl | 0.6953 | 0.7201 | 0.0248  |
| Qwen             | base     | N/A    | 0.7731 | N/A     |
| Qwen             | best_acc | 0.6503 | 0.7281 | 0.0778  |
| Qwen             | best_ppl | 0.5926 | 0.6932 | 0.1006  |
| Qwen-Instruct    | base     | N/A    | 0.8089 | N/A     |
| Qwen-Instruct    | best_acc | 0.6721 | 0.7079 | 0.0358  |
| Qwen-Instruct    | best_ppl | 0.6886 | 0.7016 | 0.013   |

### ARC_CHALLENGE

| 模型               | 配置       | 微调前    |    微调后 | 变化      |
|:-----------------|:---------|:-------|-------:|:--------|
| Llama            | base     | N/A    | 0.5324 | N/A     |
| Llama            | best_acc | 0.3951 | 0.448  | 0.0529  |
| Llama            | best_ppl | 0.3797 | 0.4317 | 0.052   |
| Llama-Instruct   | base     | N/A    | 0.5674 | N/A     |
| Llama-Instruct   | best_acc | 0.4454 | 0.4676 | 0.0222  |
| Llama-Instruct   | best_ppl | 0.4369 | 0.4744 | 0.0375  |
| Mistral          | base     | N/A    | 0.523  | N/A     |
| Mistral          | best_acc | 0.3848 | 0.4317 | 0.0469  |
| Mistral          | best_ppl | 0.3003 | 0.3942 | 0.0939  |
| Mistral-Instruct | base     | N/A    | 0.5887 | N/A     |
| Mistral-Instruct | best_acc | 0.5094 | 0.4804 | -0.029  |
| Mistral-Instruct | best_ppl | 0.4258 | 0.4497 | 0.0239  |
| Qwen             | base     | N/A    | 0.5102 | N/A     |
| Qwen             | best_acc | 0.407  | 0.4787 | 0.0717  |
| Qwen             | best_ppl | 0.3933 | 0.4556 | 0.0623  |
| Qwen-Instruct    | base     | N/A    | 0.5503 | N/A     |
| Qwen-Instruct    | best_acc | 0.4198 | 0.4744 | 0.0546  |
| Qwen-Instruct    | best_ppl | 0.4565 | 0.4514 | -0.0051 |

### OPENBOOKQA

| 模型               | 配置       | 微调前   |   微调后 | 变化     |
|:-----------------|:---------|:------|------:|:-------|
| Llama            | base     | N/A   | 0.45  | N/A    |
| Llama            | best_acc | 0.408 | 0.414 | 0.006  |
| Llama            | best_ppl | 0.356 | 0.404 | 0.048  |
| Llama-Instruct   | base     | N/A   | 0.432 | N/A    |
| Llama-Instruct   | best_acc | 0.382 | 0.4   | 0.018  |
| Llama-Instruct   | best_ppl | 0.376 | 0.398 | 0.022  |
| Mistral          | base     | N/A   | 0.442 | N/A    |
| Mistral          | best_acc | 0.372 | 0.38  | 0.008  |
| Mistral          | best_ppl | 0.366 | 0.402 | 0.036  |
| Mistral-Instruct | base     | N/A   | 0.472 | N/A    |
| Mistral-Instruct | best_acc | 0.436 | 0.422 | -0.014 |
| Mistral-Instruct | best_ppl | 0.41  | 0.41  | 0.0    |
| Qwen             | base     | N/A   | 0.47  | N/A    |
| Qwen             | best_acc | 0.354 | 0.398 | 0.044  |
| Qwen             | best_ppl | 0.364 | 0.4   | 0.036  |
| Qwen-Instruct    | base     | N/A   | 0.484 | N/A    |
| Qwen-Instruct    | best_acc | 0.416 | 0.422 | 0.006  |
| Qwen-Instruct    | best_ppl | 0.398 | 0.426 | 0.028  |

