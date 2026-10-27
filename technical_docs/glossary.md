# Glossary of Technical Terms

Plain-English definitions for the terms used in the presentation appendix and in `project_answers.md`. Each entry gives the idea first, then what it means in this project.

## Data preparation

| Term | Plain-English meaning | In this project |
| --- | --- | --- |
| **Duplicate rows** | The same record appearing more than once. | 215 of the 439 raw rows were copies; removing them left 224 campaigns. |
| **Data leakage** | Using information to make a prediction that wouldn't be available at the time the prediction is made. It makes a model look better than it can really be. | `delivery` is only known months after funding closes, so it was kept out of the models. |
| **Missing values / imputation** | Blank cells, and filling them with a sensible value. | 7 blanks, filled with the most common value (0) or, in the GAM, dropped. |
| **Log transform** | Replacing a number with its logarithm, so that proportional differences count equally ($1,000 -> $2,000 counts the same as $50,000 -> $100,000). It tames very lopsided data. | Applied to the funding goal and to the target, as log(1 + fraction raised). |
| **Skewness** | How lopsided a distribution is. 0 is symmetric; large positive values mean a long right tail. | Fraction raised has skew 4.7 (a few huge successes); after the log it's 1.3. |
| **Dummy variable** | A 0/1 column marking membership of a category. One category is left out as the *baseline* that others are compared to. | State is coded as California and New York dummies, with "Other" as the baseline. |

## Models

| Term | Plain-English meaning | In this project |
| --- | --- | --- |
| **Regression** | A model that predicts a number (here, how much of its goal a campaign raises) from other information. | All four models are regression models. |
| **OLS (ordinary least squares) regression** | Draws the "best straight-line" relationship between each factor and the outcome, adding the effects up like a scorecard. | Model 1, and the model we selected. |
| **Coefficient** | The size of one factor's effect in a regression, holding the others fixed. | Pre-order perk: +0.352 on the log scale, i.e. +42%. |
| **GAM (generalised additive model)** | A regression that lets each effect bend into a smooth curve instead of a straight line. | Model 2. The curve for goal size came out straight. |
| **Spline / knots** | The flexible curve a GAM uses, built from joined pieces. Knots are the joints; more knots allow a wigglier curve. | Tuned between 5 and 12 knots; 5 chosen. |
| **Decision tree** | A flowchart of yes/no questions ("pre-order perk? smart device?") that ends in a prediction. | The building block of models 3 and 4. |
| **Random forest** | Hundreds of decision trees, each trained on a random sample of the data, whose predictions are averaged. | Model 3, with 300 trees. |
| **Gradient boosting** | Small decision trees built one after another, each correcting the errors of the ones before. | Model 4. |
| **Interaction** | When one factor's effect depends on another (e.g. pre-orders helping tech campaigns more than others). | Tested in OLS specification 4 (p = 0.018); trees can find interactions automatically. |
| **Baseline model** | The simplest possible prediction, such as always guessing the average. A model is only useful if it beats this. | Predicting the training-fold mean gives RMSE 0.685; the models reach 0.645-0.654. |

## Fitting and tuning

| Term | Plain-English meaning | In this project |
| --- | --- | --- |
| **Overfitting** | A model learning the quirks of its training data (memorising) rather than general patterns, so it does well on data it has seen and badly on new data. | An untuned random forest explained 74% of training data and did worse than guessing on new data. |
| **Hyperparameters** | Settings that control how complex a model can be, chosen before training (as opposed to coefficients, which are learned). | Tree depth, leaf size, learning rate, number of knots, smoothing penalty. |
| **Tuning** | Trying many hyperparameter settings and keeping the one that predicts best on held-out data. | 45 combinations for the GAM and random forest; 12 random combinations per fold for gradient boosting. |
| **Grid search / randomised search** | Tuning by trying every combination in a list (grid), or a random sample of them (randomised). | Random forest and GAM: grid. Gradient boosting: randomised. |
| **Regularisation / smoothing penalty** | A built-in brake on complexity: the model pays a cost for wiggly curves or large effects. | The GAM's λ; λ = 1000 (heavy smoothing) was chosen. |
| **Tree depth** | How many yes/no questions a tree can ask in a row. Deeper = more complex. | Shallow trees won: depth 2 (random forest) and depth 1 (gradient boosting) were chosen most often. |
| **Learning rate** | In gradient boosting, how much each new tree is allowed to correct the previous ones. Smaller = more cautious. | 0.01 (the most cautious option) was chosen most often. |
| **Effective degrees of freedom (edf)** | How flexible a fitted curve really is. About 1-2 means a straight line; higher means more bends. | GAM goal-size curve: edf 2.03, i.e. a straight line. |

## Evaluation

| Term | Plain-English meaning | In this project |
| --- | --- | --- |
| **Train / test split** | Fitting a model on some data and checking it on data it never saw. | The basis of every accuracy number reported. |
| **Cross-validation (k-fold)** | Splitting the data into k parts, training on k-1 and testing on the one left out, and repeating so every part is tested once. Repeating with new random splits gives a more stable average. | 5-fold everywhere; repeated 2-10 times depending on the model. |
| **Nested cross-validation** | Cross-validation inside cross-validation. The inner loop chooses the settings, using training data only; the outer loop then scores the chosen model on untouched data. It stops tuning from flattering the results. | Used for the random forest and gradient boosting. |
| **R² (R-squared)** | The share of the variation in outcomes the model explains. 1 = perfect; 0 = no better than guessing the average; below 0 = worse than guessing. | About 0.19 in-sample, 0.04-0.08 on new campaigns. |
| **Out-of-sample / in-sample** | Performance on data the model hasn't seen vs data it was trained on. Out-of-sample is the honest measure. | In-sample R² ~0.20 vs out-of-sample ~0.05. |
| **MAE (mean absolute error)** | The average size of a model's misses, ignoring direction. Lower is better. | About 0.51 on the log scale for all four models. |
| **RMSE (root mean squared error)** | Like MAE, but big misses count extra. Lower is better. | 0.645-0.654 on the log scale; 0.685 for the no-model baseline. |
| **Permutation importance** | How much a model's accuracy on new data drops when one factor's values are randomly shuffled. A big drop means the model relies on that factor. | Pre-order perk, IoT and California had the largest drops in both tree models. |
| **Impurity (built-in) importance** | A tree model's own importance score, based on how often a factor is used to split. It favours continuous factors with many possible split points. | It ranked goal size high, but on held-out data goal size didn't help, which is why we report permutation importance. |
| **Drop-one importance** | Refit the model without a factor and see how much accuracy falls. | Used for the GAM; it agrees with the other models' rankings. |

## Statistical inference

| Term | Plain-English meaning | In this project |
| --- | --- | --- |
| **p-value** | Roughly, how surprising the result would be if the factor truly had no effect. Below 0.05 is the usual bar for "statistically significant". | Pre-order p < 0.001; IoT 0.012; California 0.011; goal size 0.063 (borderline). |
| **Confidence interval (95%)** | The range of plausible values for an effect. If it excludes 0, the effect is significant at the 5% level. | Pre-order coefficient: 0.166 to 0.538. |
| **Statistically significant** | Unlikely to be explained by chance alone. It's not the same as large or important. | Three drivers are significant; the rest are not. |
| **Heteroskedasticity** | When prediction errors are bigger for some kinds of campaigns than others. It makes ordinary p-values unreliable. | Detected (Breusch-Pagan p = 0.0046). |
| **HC3 robust standard errors** | A correction that keeps p-values and confidence intervals honest when error sizes vary. HC3 is the version recommended for small samples. | Used for the OLS results. |
| **Multicollinearity / VIF** | When two factors carry nearly the same information, effects become hard to separate. VIF measures it; above 5-10 is a concern. | All VIFs below 3.1: not a problem. |
| **AIC / BIC** | Scores that balance how well a model fits against how complex it is. Lower is better. | Used to compare the four OLS specifications. |
| **Adjusted R²** | R² with a penalty for each extra variable, so adding useless factors doesn't look like progress. | 0.155 for the primary OLS model. |
| **Correlation vs causation** | Two things moving together doesn't prove one causes the other. | Our results are associations; the recommendations should be piloted. |
