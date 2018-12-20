## 脚本功能
1. tensorboard_cnn_lr.py: 使用学习率衰减策略 （CNN）
2. tensorboard_cnn.py: 不使用学习率衰减策略，固定学习率为 0.001（CNN）
3. tensorboard_nn_lr.py: 使用学习率衰减策略（NN）
4. tensorboard_nn.py: 不适用学习率衰减策略，固定学习率为 0.001（NN）
5. tensorboard_nn_sdropout.py: 寻找最佳的 keep_prob （NN）
6. tensorboard_cnn_sdropout.py: 寻找最佳的 keep_prob （CNN）


## 主要工作
1. 建立普通神经网络模型（30%）
2. 建立卷积神经网络模型 （30%）
3. 分别寻找普通神经模型和卷积神经网络模型的最优的 keep_prob 值
4. 给两个模型都加入学习率衰减策略，看看是否能提升模型的性能
5. 使用 Tensorboard 汇总信息（主要包括 Loss，Accuracy）（20%）
6. 使用 Tensorboard 查看图片，主要是验证输入是否正确
7. 加入 batch shuffle 加快收敛（也就是每个 batch 的数据都打乱一下）
8. 数据归一化

