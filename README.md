# Mental Health Recovery Trajectories

**A university research showcase exploring early forum language, sentiment-based labels and their agreement with human judgment.**

This repository demonstrates research I completed as part of my **Master of Artificial Intelligence and Machine Learning at Adelaide University**, through university research work and assessed assignments. It presents the **Part A exploratory notebook** and a summary of the **final Part B findings**.

> [!IMPORTANT]
> **Academic demonstration only. The underlying research dataset is proprietary and is not supplied or licensed for public use.** The Part B report, assignment materials, implementation and annotation files remain private.
>
> **The repository materials must not be copied, republished, redistributed or submitted as someone else's work, in whole or in part, on any website, repository, platform or assignment.** No new permission for those uses is granted here. The scope and necessary exceptions for earlier licences, GitHub's terms and applicable law are explained under [Usage restrictions](#usage-restrictions).

[View the Part A notebook](Mental_Health_Trajectories_Part_A.ipynb) · [Read the Part B results](#part-b-results) · [Portfolio case study](https://www.hariprasad.world/mental-health-research)

## Research question and main finding

**Can language in the first five comments of a discussion thread help predict the original poster's later trajectory?**

I investigated this through data preparation, linguistic feature engineering, sentiment-based labelling and supervised classification. The project then tested a more fundamental question: whether those automatically generated labels captured what a human reader judged to be improvement, stability or worsening.

**The central finding was weak agreement between the sentiment-derived labels and human annotation.** On a 60-thread reference set, VADER achieved Cohen's **κ = 0.200**; an SVM trained on VADER labels achieved **κ = -0.020** against the human annotations. Predicting an automated sentiment category did not establish reliable prediction of recovery.

The labels `Improved`, `Stable` and `Worsened` are experimental categories. They are **not clinical diagnoses, verified recovery outcomes or validated risk assessments**.

## What I built

- A pipeline to ingest category-specific forum files, investigate identifiers and reconstruct thread interactions.
- A separation between early comments used as predictors and later comments used to construct the target.
- A Part A Random Forest baseline using first-person pronoun counts.
- A Part B comparison using behavioural features, TF-IDF, Logistic Regression, Random Forest and Linear SVM.
- A blinded human-annotation exercise, agreement analysis and an evaluation of models against those annotations.

## At a glance

Both stages used source files containing **4,735 post rows and 306,334 comment rows across nine categories**. Their modelling datasets and targets differ:

| | Part A: public notebook | Part B: private implementation, public results summary |
| --- | --- | --- |
| Modelling threads | 4,698 | 4,428 |
| Early input | First five comments in each thread | First five comments in each thread |
| Later target text | All comments after the first five | Only the original poster's comments after the first five |
| Features | Two pronoun counts | Five behavioural features and 150 TF-IDF features |
| Final model comparison | Random Forest | Logistic Regression, Random Forest and Linear SVM |
| Main evaluation | Stratified 80/20 thread split | Stratified 80/20 split, plus a separate 60-thread human-reference evaluation |
| Key result | 0.788 accuracy; 0.31 macro-F1 | 0.424 macro-F1 on the original Part B labels; weak human-label agreement |

These are separate experiments. Their scores should not be treated as a directly comparable before-and-after improvement.

## Part A: exploratory pipeline and baseline

The [public notebook](Mental_Health_Trajectories_Part_A.ipynb) records exploratory analysis, intermediate approaches and the initial modelling pipeline. Its code and saved results have been preserved.

### Data preparation

The final ingestion block reads category-specific `*_post.csv` and `*_comment.csv` files. It derives a forum category from each filename and joins records using a composite of **category and post ID**, addressing the ambiguity of post IDs reused across categories. The saved output records **306,334 merged interaction rows**.

The notebook also explores exact-row deduplication, UUID identifiers and nearest-preceding-post matching. These are earlier experiments: the final ingestion block reloads the source files and uses the composite-key merge. It does not carry forward all of the earlier cleaning decisions.

### Features and target

After parsing timestamps and ordering comments within each thread, the first five comments supply two features: raw counts of first-person singular and plural pronouns.

The target is the **mean VADER compound sentiment of all remaining comments**, including community replies. A score above `0.05` is labelled `Improved`, below `-0.05` is `Worsened`, and otherwise `Stable`. This measures later sentiment; it is not a measured change in the original poster's health.

### Recorded baseline

A Random Forest with **100 trees**, balanced class weights and `random_state=42` was fitted using a stratified split of **3,758 training threads and 940 test threads**.

| Metric | Saved notebook result |
| --- | ---: |
| Accuracy | 0.788 |
| Macro-F1 | 0.31 |
| Improved recall | 0.85 |
| Stable recall | 0.03 |
| Worsened recall | 0.02 |

The test set contained 865 `Improved`, 29 `Stable` and 46 `Worsened` labels. High overall accuracy therefore concealed very poor recall for the two minority classes.

## Part B results

**Part B refined the target, expanded the features and evaluated label validity.** This section summarises aggregate findings from my private final report. The public Part A notebook does not contain the Part B implementation and cannot reproduce these results.

### Changes to the experiment

- Retained **4,428 threads** with a later comment from the original poster; community replies were excluded from the target.
- Used asymmetric sentiment boundaries: `Improved` at or above `0.25`, `Worsened` at or below `-0.05`, and `Stable` otherwise.
- Combined singular/plural pronoun counts, time between the first two comments, sentiment volatility and sentiment slope with **150 TF-IDF unigram/bigram features**.
- Fitted the vectoriser and relevant feature scaling on training data, then compared three supervised models.
- Annotated **60 threads** with the automated label hidden and used them to examine agreement with human judgment.

### Classification against automated labels

The original Part B target contained 3,015 `Improved`, 682 `Stable` and 731 `Worsened` labels. The stratified 80/20 split produced **3,542 training threads and 886 test threads**.

| Model | Accuracy | Balanced accuracy | Macro-F1 |
| --- | ---: | ---: | ---: |
| Stratified random baseline | 0.497 | 0.309 | 0.309 |
| Random Forest | 0.669 | 0.359 | 0.330 |
| Logistic Regression | 0.509 | 0.460 | **0.424** |
| Linear SVM | 0.500 | 0.447 | 0.414 |

Logistic Regression had the highest macro-F1 on this target. Random Forest recalled only **1% of Stable and 12% of Worsened** cases; Linear SVM recalled **33% and 47%**, respectively.

A separate Linear SVM experiment on revised automated labels recorded **0.440 macro-F1**, **0.481 balanced accuracy** and **0.523 accuracy**. The report calls these “validated labels”, but they are not independently established clinical ground truth. The changed target distribution means this score is not a directly comparable improvement over the table above.

### Agreement with human annotation

I annotated the 60 reference threads with the automatic labels hidden. There was **one annotator**, so this provides a human reference rather than independently established ground truth.

VADER agreed with these annotations in **46.7%** of cases: **Cohen's κ = 0.200**, with a **95% bootstrap confidence interval of 0.01–0.37** from 2,000 resamples.

For the separate model evaluation, classifiers were trained on the **4,368 non-reference threads** and assessed against the same 60 human annotations:

| Predictor evaluated against human annotations | Macro-F1 | Cohen's κ |
| --- | ---: | ---: |
| VADER labeller itself | 0.461 | 0.200 |
| Linear SVM trained on VADER labels | 0.324 | -0.020 |
| Linear SVM trained on revised automated labels | 0.380 | 0.085 |

The revised labeller was selected using those same 60 reference threads. Excluding them from supervised training did **not** remove that selection bias.

Threshold tuning also exposed an evaluation problem: VADER initially reached **κ = 0.242** when thresholds were fitted and scored on the same annotations. The corrected **five-fold out-of-fold result was κ = 0.064**. The apparent gain did not survive that correction.

## Interpretation and limitations

The research contribution is the combination of a working exploratory pipeline, class-sensitive evaluation and evidence that the target labels need better validation. The experiments do not establish a universal mathematical limit on models trained with noisy labels.

- **Target validity:** sentiment, supportive language and gratitude do not necessarily indicate recovery. Author isolation reduces one source of contamination but does not establish a valid clinical target.
- **Small human reference:** 60 threads and one annotator give limited evidence, with no inter-annotator agreement estimate and bias from reusing the set for labeller selection.
- **Data quality:** the final Part B report records retained duplicate posts and unparseable timestamps. The Part A final reload also does not retain the earlier deduplication.
- **Sampling and evaluation:** requiring the original poster to return excludes other users. Data come from one platform, and the reported random thread splits do not establish generalisation to new authors, later periods or other communities.
- **Practical use:** these results do not support clinical decision-making, diagnosis, automated triage or deployment as a mental-health risk detector.

A stronger follow-up would establish a clearer annotation codebook, involve multiple annotators and evaluate against an independent human-labelled set before further model optimisation.

## Repository contents and data availability

| Material | Availability |
| --- | --- |
| Part A exploratory notebook and its saved outputs | Included for inspection as a research demonstration |
| Part B aggregate findings | Summarised in this README |
| Part B final report, assignment submissions and implementation | Private; not published here |
| Underlying proprietary dataset, source CSVs and annotation files | Not supplied or licensed for public use |

**The research dataset is proprietary and must not be copied, shared, uploaded or redistributed.** Public visibility of the source forum does not grant permission through this repository to reuse its content. Rights in forum content and third-party materials remain with their respective rights holders.

The notebook is a preserved research artefact, not a packaged application or a supported reproduction kit. Its recorded kernel is Python 3.13.3; imports include pandas, NumPy, Matplotlib, seaborn, NLTK/VADER and scikit-learn. Package versions are not pinned, and the restricted inputs prevent independent end-to-end reproduction from this repository alone.

For code inspection, the final Part A path is in the blocks headed **Importing the libraries**, **Merging the Datasets**, **Implementing VADER** and **Implement the Random Forest Classifier Model**. Earlier cells preserve development experiments, so the notebook should not be assumed to support a clean “Restart Kernel → Run All” execution. No fresh experiments were run to prepare this README.

## Usage restrictions

**Academic showcase only — all rights reserved, subject to the exceptions below.** This work was completed for university research and assignments and is displayed to demonstrate my methods, technical contribution and findings.

Unless separately authorised in writing by the relevant rights holder, repository materials must strictly **not be copied, reproduced, adapted, republished, redistributed, mirrored or uploaded elsewhere**, in whole or in part. They must not be submitted as someone else's coursework, assignment, research or original work. Attribution alone does not grant reuse permission.

The [repository rights notice](LICENSE) records these restrictions. It does not claim to revoke permissions already granted for material in earlier MIT-licensed revisions, override GitHub's public-repository viewing and forking terms, remove rights provided by applicable law, or alter third-party licences. No new reuse licence is offered by this revision.

## Author

**Hari Prasad Rangaraj** · Master of Artificial Intelligence and Machine Learning, Adelaide University

**Supervisor:** Dr Menasha Thilakaratne


*README reviewed and updated: 29 September 2026.*
