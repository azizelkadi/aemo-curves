# Papers consulted

All papers visited by the five-expert board while preparing [`research_proposals_20.md`](research_proposals_20.md). Each entry: title · year · URL · 1-line summary.

## Demand- and supply-curve forecasting (electricity-specific)

- **Sinha & Lucheroni, *Demand and supply curve forecasting using a monotonic autoencoder for short-term day-ahead electricity market bid curves*** (Applied Energy, 2025). https://doi.org/10.1016/j.apenergy.2025.126262 - Monotonic autoencoder + low-d latent forecast for IPEX-NORD day-ahead bid/offer curves; the closest analogue in the literature to this repo's problem.
- **Li, Alonso, Pascual, *Predicting electricity supply and demand curves with functional data techniques*** (IJEPES 166, 2025). https://doi.org/10.1016/j.ijepes.2025.110561 - FPCA + functional regression on OMIE day-ahead supply/demand curves, with a "closest historical curve" snap step.
- **Aneiros, Vilar, Cao, Muñoz San Roque, *Functional prediction for the residual demand in electricity spot markets*** (IEEE TPWRS 28(4), 2013). https://www.researchgate.net/publication/281149602 - Functional AR(1) on OMIE residual-demand curves; canonical reference for FAR-on-electricity.
- **Aneiros, Vilar, Raña, *Short-term forecast of daily curves of electricity demand and price*** (IJEPES 80, 2016). https://www.sciencedirect.com/science/article/abs/pii/S0142061515004810 - Semi-functional partial-linear models for daily demand / price curves; standard FDA reference.
- **Peláez, Aneiros, Vilar, *Bootstrap prediction regions for daily curves of electricity demand and price using functional data*** (arXiv, 2024). https://arxiv.org/abs/2401.11885 - Bootstrap prediction intervals around FDA point forecasts of daily electricity curves.
- **Vilar, Cao, Aneiros, *Forecasting next-day electricity demand and price using nonparametric functional methods*** (IJEPES 39, 2012). https://www.sciencedirect.com/science/article/pii/S0142061511002201 - Kernel-FDA point and probabilistic forecasts of next-day OMIE curves; foundational paper for the FDA-electricity literature.
- **Vivó & Alonso, *Prediction of Intraday Electricity Supply Curves*** (Applied Sciences 14(22), 2024). https://www.mdpi.com/2076-3417/14/22/10663 - Quantile-curve forecasting in inverse-CDF space on Spanish intraday markets.
- **Liebl, *Modeling and forecasting electricity spot prices: A functional data perspective*** (Ann. Appl. Stat., 2013). https://arxiv.org/pdf/1310.1628 - Influential FDA treatment of EEX spot-price *price-demand* curves; introduces the functional-time-series viewpoint widely cited thereafter.

## Functional data analysis (general)

- **Ramsay, *Estimating smooth monotone functions*** (JRSS B 60, 1998). https://doi.org/10.1111/1467-9868.00130 - The "monotone smoother" via integration of `exp(W(t))`; the trick behind every modern guaranteed-monotonic functional forecast.
- **Shin, Zhou, Ding, *Joint estimation of monotone curves via functional principal component analysis*** (CSDA 166, 2022). https://doi.org/10.1016/j.csda.2021.107343 - Joint FPCA reparametrisation enforcing monotonicity across a panel of curves; used by Sinha-Lucheroni.

## Probabilistic / distributional forecasting

- **Dabney, Ostrovski, Silver, Munos, *Implicit Quantile Networks for Distributional RL*** (ICML 2018). https://arxiv.org/abs/1806.06923 - Continuous quantile loss for arbitrary `τ`; basis for the IQN family.
- **Gouttes, Rasul, Koren, Stephan, Naghibi, *Probabilistic Time Series Forecasting with Implicit Quantile Networks*** (ICML TS Workshop, 2021). https://arxiv.org/abs/2107.03743 - IQN ported to time-series probabilistic forecasting; improves CRPS over MQ-RNN on electricity load.
- **Smyl et al., *Any-Quantile Probabilistic Forecasting of Short-Term Electricity Demand*** (2024). https://arxiv.org/abs/2404.17451 - Single network outputs any requested quantile of short-term load via `τ` conditioning.
- **Marcjasz, Narajewski, Weron, Ziel, *Distributional neural networks for electricity price forecasting*** (Energy Economics, 2023). https://arxiv.org/abs/2207.02832 - DDNN with Johnson's-SU output for heavy-tailed German day-ahead price distributions.
- **Lipiecki, Uniejewski, Weron, *Postprocessing of point predictions for probabilistic forecasting of day-ahead electricity prices: The benefits of using Isotonic Distributional Regression*** (Energy Economics, 2024). https://arxiv.org/abs/2404.02270 - Isotonic distributional regression as a cheap post-hoc calibration that often beats DDNN-JSU.
- **Duan et al., *NGBoost: Natural Gradient Boosting for Probabilistic Prediction*** (ICML 2020). https://arxiv.org/abs/1910.03225 - Boosted-tree distributional regression with natural-gradient updates; CRPS-trainable.
- **Uniejewski & Weron, *Regularized quantile regression averaging for probabilistic electricity price forecasting*** (Energy Economics, 2021). https://www.sciencedirect.com/science/article/abs/pii/S0140988321000268 - Lasso-regularised QRA - strong tabular benchmark on EPF.
- **Mastropietro et al., *On-line conformalized neural networks ensembles for probabilistic forecasting of day-ahead electricity prices*** (Applied Energy, 2024). https://arxiv.org/abs/2404.02722 - Online conformal calibration on NN ensembles for EPF; tight marginal coverage.
- **Cassina et al., *Conformal Prediction for Electricity Price Forecasting in the Day-Ahead and Real-Time Balancing Market*** (2025). https://arxiv.org/abs/2502.04935 - Conformal-prediction calibration applied to DA and balancing-market price forecasts.

## Deep learning for time-series forecasting

- **Nie et al., *A Time Series is Worth 64 Words: Long-term Forecasting with Transformers (PatchTST)*** (ICLR 2023). https://arxiv.org/abs/2211.14730 - Patch-based transformer for long-horizon univariate forecasting; established SOTA on M4-style benchmarks.
- **Liu et al., *iTransformer: Inverted Transformers Are Effective for Time Series Forecasting*** (ICLR 2024). https://arxiv.org/abs/2310.06625 - Tokenise each *variate* (not each timestep); attention then learns cross-variate dependencies.
- **Challu et al., *N-HiTS: Neural Hierarchical Interpolation for Time Series Forecasting*** (AAAI 2023). https://arxiv.org/abs/2201.12886 - Hierarchical multi-rate MLP blocks for long-horizon forecasting; cheap and accurate.
- **Liu et al., *Retrieval-Augmented Diffusion Models for Time Series Forecasting (RATD)*** (NeurIPS 2024). https://proceedings.neurips.cc/paper_files/paper/2024/file/053ee34c0971568bfa5c773015c10502-Paper-Conference.pdf - Diffusion forecasting conditioned on retrieved nearest-neighbour series; meaningful gains on energy benchmarks.
- **Rasul et al., *TimeGrad: Autoregressive Denoising Diffusion Models for Multivariate Probabilistic Time Series Forecasting*** (ICML 2021). https://arxiv.org/abs/2101.12072 - Score-based diffusion for multivariate probabilistic time-series forecasting.
- **Tashiro et al., *CSDI: Conditional Score-based Diffusion Models for Probabilistic Time Series Imputation*** (NeurIPS 2021). https://arxiv.org/abs/2107.03502 - Diffusion-based probabilistic imputation that doubles as a forecaster.
- **Ansari et al., *Chronos: Learning the Language of Time Series*** (2024). https://arxiv.org/abs/2403.07815 - Pretrained TS foundation model (AWS); strong zero-shot results across diverse benchmarks.
- **Hertel et al., *Benchmarking Time Series Foundation Models for Short-Term Household Electricity Load Forecasting*** (2024). https://arxiv.org/abs/2410.09487 - Empirical comparison of Chronos / TimesFM / Lag-Llama on residential load.
- **Meyer et al., *Time Series Foundation Models for Energy Load Forecasting on Consumer Hardware: A Multi-Dimensional Zero-Shot Benchmark*** (2026). https://arxiv.org/abs/2602.10848 - Zero-shot TSFM benchmark on ERCOT load showing ~47 % MASE reduction vs seasonal-naïve.
- **Lim, Arik, Loeff, Pfister, *Temporal Fusion Transformers for interpretable multi-horizon time series forecasting*** (Int. J. Forecasting, 2021). https://www.sciencedirect.com/science/article/pii/S0169207021000637 - TFT: gating + multi-head attention for multi-horizon forecasting with mixed static / past / known-future covariates.

## Representation / metric learning

- **Yue et al., *TS2Vec: Towards Universal Representation of Time Series*** (AAAI 2022). https://arxiv.org/abs/2106.10466 - Hierarchical contrastive learning of dense time-series embeddings; strong on retrieval and classification.
- **Tonekaboni, Eytan, Goldenberg, *Unsupervised Representation Learning for Time Series with Temporal Neighborhood Coding (TNC)*** (ICLR 2021). https://arxiv.org/abs/2106.00750 - Self-supervised TS embeddings using stationarity-defined temporal neighbourhoods as positives.
- **Chen, Kornblith, Norouzi, Hinton, *A Simple Framework for Contrastive Learning of Visual Representations (SimCLR)*** (ICML 2020). https://arxiv.org/abs/2002.05709 - Canonical NT-Xent contrastive learner; template for #17 and most subsequent SSL papers.
- **Khosla et al., *Supervised Contrastive Learning*** (NeurIPS 2020). https://arxiv.org/abs/2004.11362 - Supervised positive sets within contrastive loss; outperforms cross-entropy on many tasks.
- **Frogner, Mirzazadeh, Solomon, *Learning Embeddings into Entropic Wasserstein Spaces*** (ICLR 2019). https://openreview.net/forum?id=rJg4J3CqFm - Train an encoder so that L² in the embedding space matches Wasserstein in the data space - direct theoretical motivation for Proposal #18.
- **Schroff, Kalenichenko, Philbin, *FaceNet: A Unified Embedding for Face Recognition and Clustering*** (CVPR 2015). https://arxiv.org/abs/1503.03832 - Triplet-loss metric learning that defined the modern recipe.
- **Lee et al., *Set Transformer*** (ICML 2019). https://arxiv.org/abs/1810.00825 - Permutation-invariant transformer over sets via inducing-point attention; useful for variable-length step curves.
- **Zaheer et al., *Deep Sets*** (NeurIPS 2017). https://arxiv.org/abs/1703.06114 - Theoretical foundation: any permutation-invariant function decomposes as `ρ(Σ φ(x))`.
- **Huang et al., *Learning Deep Structured Semantic Models for Web Search using Clickthrough Data (DSSM)*** (CIKM 2013). https://www.microsoft.com/en-us/research/publication/learning-deep-structured-semantic-models-for-web-search-using-clickthrough-data/ - Two-tower bi-encoder retrieval; pattern reused everywhere in modern dense retrieval.
- **Nogueira, Yang, Cho, Lin, *Multi-Stage Document Ranking with BERT*** (2019). https://arxiv.org/abs/1910.14424 - Retrieve-then-rerank with a cross-encoder; the recipe behind almost every modern IR system.

## Wasserstein / optimal-transport autoencoders

- **Tolstikhin et al., *Wasserstein Auto-Encoders*** (ICLR 2018). https://arxiv.org/abs/1711.01558 - Optimal-transport-based autoencoder objective; alternative to VAEs.
- **Patrini et al., *Sinkhorn AutoEncoders*** (UAI 2019). https://www.auai.org/uai2019/proceedings/papers/253.pdf - Sinkhorn-divergence reconstruction + prior matching; differentiable Wasserstein AE.
- **Kolouri et al., *Sliced-Wasserstein Autoencoder*** (ICLR 2019). https://arxiv.org/abs/1804.01947 - Cheap sliced-Wasserstein approximation as the autoencoder's distributional loss.

## Structural / market microstructure

- **Coulon & Howison, *Stochastic Behaviour of the Electricity Bid Stack*** (2009). https://people.maths.ox.ac.uk/howison/papers/CoulonHowison211008.pdf - Parametric stochastic model of the supply bid stack; basis for the structural-bid-stack literature.
- **Carmona, Coulon, Schwarz, *Electricity price modeling and asset valuation: a multi-fuel structural approach*** (2013). https://arxiv.org/pdf/1205.2299 - Multi-fuel structural model linking fuel prices to electricity spot prices via the bid stack.
- **Carmona & Coulon, *A Survey of Commodity Markets and Structural Models for Electricity Prices*** (2014). https://www.researchgate.net/publication/264890880 - Survey of structural / fundamentals-based electricity-price models; canonical reference.

## Australian electricity market / WA / NEM

- **Bell et al., *The duck curve in a drying pond: The impact of rooftop PV on the Western Australian electricity market transition*** (Energy Research & Social Science, 2021). https://www.sciencedirect.com/science/article/abs/pii/S0957178721000667 - Empirical analysis of how rooftop PV reshaped the WA SWIS load curve into a "duck curve".
- ***The impact of rooftop solar on wholesale electricity demand in the Australian NEM*** (Frontiers in Energy Research, 2023). https://www.frontiersin.org/articles/10.3389/fenrg.2023.1197504/full - Quantifies how rooftop PV in the NEM lowers wholesale daytime demand by 10-30 %.

## Load forecasting / weather sensitivity

- **Hekkenberg et al., *Indications for a changing electricity demand pattern: The temperature dependence of electricity demand in the Netherlands*** (Energy Policy, 2009). - Documents the non-linear T-vs-load relationship (heating + cooling threshold effects); template for temperature-conditioned forecasters.
- ***Seasonal forecasting of hourly electricity demand*** (Nature Scientific Reports, 2025). https://www.nature.com/articles/s41598-025-91878-0 - Seasonal-scale hourly load forecasting combining weather and calendar features.
