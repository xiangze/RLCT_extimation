# LLC (Local leaning coefficient, Real Log Canoical Threshold) estimation

from [RLCT repository](https://github.com/suswei/RLCT)

## code
- estimate_softmaxDNN.py

compare exact RCLT(Learning Coeciffient) of linear NN and nonlinear NN (softmax/Relu)

- llt_rlct_estimator_for_softmax_DNN.py

    RLCT of small nonlinear coefficient to compare linear matrix decomposition network(exact solutoin).

- llc_rlct_estimator_for_softmax_networks_via_power_posteriors_sgld.py

    outor interpolation of SGLD LLC extimation with NUTS/HMC SGLD extimaiton with pyto.

- llc_training_experiment.py

    LLT selection obserbation during SGD update, small LLC saddlepoint is selected or not.

## how to  run
```
python estimate_softmaxDNN.py
```

# Reference
- [The Local Learning Coefficient: A Singularity-Aware Complexity Measure](https://arxiv.org/abs/2308.12108)
- [Differentiation and Specialization of Attention Heads via the Refined Local Learning Coefficient](https://arxiv.org/abs/2410.02984)
- [Quantifying degeneracy in singular models via the learning coefficient](https://www.researchgate.net/publication/373332996_Quantifying_degeneracy_in_singular_models_via_the_learning_coefficient)
- [Estimating the Local Learning Coefficient](https://danmackinlay.name/notebook/estimating_llc.html)
- [Singular Learning Theory with Daniel Murfet](https://axrp.net/episode/2024/05/07/episode-31-singular-learning-theory-dan-murfet.html)
- [ReLUネットワークにおける局所学習係数推定手法のモデル選択への応用](https://www.jstage.jst.go.jp/article/pjsai/JSAI2025/0/JSAI2025_1S5GS201/_pdf/-char/en)
- https://arxiv.org/abs/1806.09597
- https://arxiv.org/abs/1704.04289
- https://github.com/suswei/RLCT
- https://github.com/edmundlth/scalable_learning_coefficient_with_sgld/tree/v1.0MultiparamShapes.py

