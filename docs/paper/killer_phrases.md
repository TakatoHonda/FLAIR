# FLAIR Paper: Killer Phrases & Key Narratives

## Title Candidates

- "FLAIR: 6 Parameters Beat 46 Million"
- "Periodic Time Series Are Rank-1 Matrices — And That's All You Need"
- "One SVD to Forecast Them All: How a 6-Parameter Statistical Method Matches Foundation Models"
- "FLAIR: Factored Level And Interleaved Ridge for Time Series Forecasting"

## Abstract Openers

- "We show that a statistical method with 6 parameters, one SVD, and no GPU matches Chronos-small (46M parameters) on the GIFT-Eval benchmark."
- "Why do Foundation Models with billions of parameters lose to a method with one SVD and zero hyperparameters?"

## Killer Numbers

- **6 vs 46,000,000**: FLAIR's parameters vs Chronos-small. 7.7 million x difference.
- **relMASE ~0.89**: matches Chronos-small (0.892), beats N-BEATS (0.938), TFT (0.915), DeepAR (1.343)
- **29 minutes on CPU**: no GPU, no training loop, no pre-training data
- **Zero hyperparameters**: everything is data-driven (alpha via GCV-SA, kappa via MoM, P via MDL)
- **One SVD**: the entire model fits in one matrix decomposition

## Killer Phrases (for paper body)

### The Core Insight
- "A periodic time series, when reshaped by its period, is approximately a rank-1 matrix."
- "This is not a hypothesis — it is a structural property that holds empirically across 97 configurations."

### On Simplicity
- "The entire method is one equation: y_hat(phase, period) = Level_hat(period) x Shape(phase)."
- "FLAIR has fewer parameters per series (~6) than most neural networks have layers."
- "Everything in FLAIR is closed-form. There are no gradients, no epochs, no learning rate schedules."

### On the Frozen Shape Theorem
- "We tested 17 different approaches to make Shape adaptive. Every single one made things worse."
- "Shape should not be learned. The K=5 average of raw proportions is optimal — a minimum-variance unbiased estimator at this sample size."
- "This is FLAIR's deepest empirical finding: do not try to improve the Shape."

### On Dirichlet Shape
- "We don't choose between weekday and weekend patterns — the Dirichlet posterior automatically blends them based on data availability."
- "When data is scarce, the posterior converges to the global Shape. When data is abundant, it specializes. There is no threshold, no if/else — just Bayes' rule on the simplex."

### On MDL Period Selection
- "FLAIR doesn't detect periods from spectral analysis — it asks: for which period does the rank-1 assumption hold best?"
- "Period discovery as minimum-description-length selection over periodic low-rank matrixizations."
- "The SVD itself tells us which period is correct, without running the full pipeline."

### On Location Shift
- "A single line of code — y += max(1 - min(y), 1) — reduced relMASE from 0.983 to 0.920. The root cause of ETT2's catastrophic failure was not non-stationarity but a preprocessing bug that destroyed 55% of the data."

### On Beating Foundation Models
- "6 parameters. No GPU. No pre-training corpus. No attention mechanism. Yet FLAIR matches Chronos-small on 97 configurations across 23 datasets."
- "The gap between statistical methods and Foundation Models is not as large as the literature suggests — it is largely a gap in engineering, not in modeling."
- "FLAIR demonstrates that the rank-1 structure of periodic time series is a stronger inductive bias than billions of parameters."

### On Why This Works (Three Compressions)
- "FLAIR achieves three simultaneous compressions: n→n/P samples (noise reduction by sqrt(P)), H→ceil(H/P) forecast steps (24x less error accumulation), and structural Shape (overfitting immunity)."
- "When Ridge penalty alpha tends to infinity, FLAIR degenerates to Seasonal Naive — the strongest simple baseline. Standard Ridge degenerates to zero."

### For Rebuttal / Related Work
- "Unlike SparseTSF (ICML 2024), FLAIR uses the rank-1 structure for both decomposition AND forecasting, not just as a linear layer architecture."
- "Unlike DLinear, FLAIR's decomposition is multiplicative and period-aligned, not additive with a moving average."
- "FLAIR is the first method to apply Dirichlet-Multinomial shrinkage to temporal disaggregation proportions."
- "MDL-based period selection from the SVD spectrum is, to our knowledge, novel in the time series forecasting literature."

## Key Figures for Paper

1. **Rank-1 visualization**: heatmap of reshaped matrix + SVD singular values (existing fig1)
2. **Benchmark scatter plot**: relMASE vs relCRPS, FLAIR between FMs and statistical methods (existing fig4, updated)
3. **Frozen Shape bar chart**: 17 failed attempts (existing fig3)
4. **Ablation table**: V9 → +Dirichlet → +Location Shift → +MDL Period → each improvement
5. **Parameter efficiency plot**: relMASE vs log(parameters), FLAIR as extreme outlier
6. **Per-dataset heatmap**: FLAIR vs top 5 methods across all 97 configs

## Paper Structure (Suggested)

1. Introduction (the puzzle: why does this work?)
2. The Observation (rank-1 structure)
3. Method
   - 3.1 Level x Shape decomposition
   - 3.2 Soft-average GCV Ridge
   - 3.3 Dirichlet Shape with calendar context
   - 3.4 MDL period selection
   - 3.5 Location-shift Box-Cox
4. Why This Works (three compressions, Frozen Shape theorem)
5. Experiments
   - 5.1 GIFT-Eval benchmark
   - 5.2 Ablation study
   - 5.3 Comparison with Foundation Models
6. Related Work
7. Conclusion

## Target Venues

- **NeurIPS 2026** (deadline ~May): strongest if relMASE < 0.89
- **ICML 2026**: similar bar
- **AAAI 2027**: slightly lower bar, good fit for "simple methods that work"
- **IJF** (International Journal of Forecasting): natural home, forecasting community values simplicity
