Hypothesis 1: Comparisons of pre and post experiment questionnaires will demonstrate a significant difference in trust scores for overall trust as aggregate measures well as emotional and analytical trust aggregate measures individually. 


\begin{enumerate}
\item A paired-samples $t$-test was conducted to determine if there was a significant difference between Total Trust (Pre) and Total Trust (Post). Results indicated that Total Trust (Pre) ($M = 4.73$, $SD = 12.32$) was significantly higher than Total Trust (Post) ($M = 1.06$, $SD = 9.56$), $t(503) = 10.39$, $p < .001$, $d = 0.46$.
\item A paired-samples $t$-test was conducted to determine if there was a significant difference between Total Emotional Trust (Pre) and Total Emotional Trust (Post). Results indicated that Total Emotional Trust (Pre) ($M = 2.74$, $SD = 4.31$) was significantly higher than Total Emotional Trust (Post) ($M = 0.40$, $SD = 0.92$), $t(503) = 12.80$, $p < .001$, $d = 0.57$.
\item A paired-samples $t$-test was conducted to determine if there was a significant difference between Total Analytical Trust (Pre) and Total Analytical Trust (Post). Results indicated that Total Analytical Trust (Pre) ($M = 1.99$, $SD = 9.44$) was significantly higher than Total Analytical Trust (Post) ($M = 0.66$, $SD = 9.24$), $t(503) = 4.85$, $p < .001$, $d = 0.22$.
\end{enumerate}

Hypothesis 2: There will be a significantly greater pre-post trust difference in the interactive condition compared to the static condition (operationalized as Text in this dataset).

Evaluation checklist (Hypothesis 2):
1. Prompt to self: Are condition labels valid for this hypothesis comparison?
Answer: Yes. The dataset contains Interactive and Text, and Text is used as the static comparison condition.
2. Prompt to self: Is the test relevant to a between-condition change hypothesis?
Answer: Yes. Welch's independent-samples $t$-test on change scores directly tests Interactive vs Text mean change differences.
3. Prompt to self: Is there a robustness check for non-normality and ordinal noise?
Answer: Yes. Mann-Whitney U was run for each metric as a non-parametric sensitivity check.
4. Prompt to self: Do descriptive means point in the same direction as the inferential result?
Answer: Yes. Means are very close between conditions for all three metrics, consistent with non-significant tests.
5. Prompt to self: Do the results make substantive sense with the observed effect sizes?
Answer: Yes. Cohen's $d$ values are near zero ($-0.00$, $0.04$, $-0.03$), indicating negligible condition effects.

\begin{enumerate}
\item A Welch's independent-samples $t$-test was conducted to determine if there was a significant difference in Overall Trust Change between Interactive and Text conditions. Results indicated that Interactive ($M = -3.68$, $SD = 8.21$) was not significantly different from Text ($M = -3.65$, $SD = 7.64$), $t(499.41) = -0.04$, $p = .964$, $d = -0.00$.
\item A Welch's independent-samples $t$-test was conducted to determine if there was a significant difference in Emotional Trust Change between Interactive and Text conditions. Results indicated that Interactive ($M = -2.26$, $SD = 4.05$) was not significantly different from Text ($M = -2.41$, $SD = 4.16$), $t(501.64) = 0.41$, $p = .680$, $d = 0.04$.
\item A Welch's independent-samples $t$-test was conducted to determine if there was a significant difference in Analytical Trust Change between Interactive and Text conditions. Results indicated that Interactive ($M = -1.42$, $SD = 6.42$) was not significantly different from Text ($M = -1.24$, $SD = 5.88$), $t(498.16) = -0.33$, $p = .739$, $d = -0.03$.
\end{enumerate}

Decision for Hypothesis 2: Not supported.

Hypothesis 2a (additional analysis): Within each condition independently, pre-post trust scores will be tested to evaluate condition-specific simple effects.

Evaluation checklist (Hypothesis 2a):
1. Prompt to self: Is this analysis aligned with a split-by-condition follow-up to Hypothesis 2?
Answer: Yes. It directly tests pre-post change within Interactive and within Text separately.
2. Prompt to self: Is the inferential method appropriate for within-condition pre-post change?
Answer: Yes. Paired-samples $t$-tests were used within each condition, with Wilcoxon signed-rank robustness checks.
3. Prompt to self: Do both conditions show statistically reliable pre-post overall trust change?
Answer: Yes. Both Interactive and Text show significant pre-post declines in overall trust ($p < .001$).
4. Prompt to self: Is the magnitude of change descriptively larger in one condition?
Answer: Yes, but only trivially. Absolute mean overall change is slightly larger in Interactive ($3.68$) than Text ($3.65$).
5. Prompt to self: Does this overturn the primary between-condition conclusion from Hypothesis 2?
Answer: No. It supports strong within-condition change in both conditions, but does not establish a significant between-condition difference in change.

\begin{enumerate}
\item A paired-samples $t$-test was conducted to determine if there was a significant difference between pre and post Overall Trust scores within the Interactive condition. Results indicated that post scores ($M = 1.20$, $SD = 9.07$) were significantly lower than pre scores ($M = 4.88$, $SD = 11.68$), $t(251) = 7.12$, $p < .001$, $d = -0.45$.
\item A paired-samples $t$-test was conducted to determine if there was a significant difference between pre and post Overall Trust scores within the Text condition. Results indicated that post scores ($M = 0.92$, $SD = 10.04$) were significantly lower than pre scores ($M = 4.58$, $SD = 12.95$), $t(251) = 7.59$, $p < .001$, $d = -0.48$.
\item Supplementary simple-effects checks for Emotional and Analytical trust were also significant within both conditions (all paired $t$-tests $p < .01$).
\end{enumerate}

Decision for Hypothesis 2a: Supported for condition-specific pre-post change in both Interactive and Text; this does not change the non-significant between-condition result in Hypothesis 2.

Hypothesis 3: The WEIRD group will demonstrate no significant difference in the change of pre-post trust scores when comparing emotional versus analytical trust.

Evaluation checklist (Hypothesis 3):
1. Prompt to self: Are emotional and analytical changes on comparable scales?
Answer: Not naturally. Emotional and analytical totals use different ranges, so both were converted to z-scored change values before paired comparison.
2. Prompt to self: Is the test relevant to "across trust type" comparison within the same participants?
Answer: Yes. A paired-samples $t$-test compares emotional vs analytical change within WEIRD-like individuals.
3. Prompt to self: Is there a robustness check?
Answer: Yes. Wilcoxon signed-rank test was run as a non-parametric paired check.
4. Prompt to self: Do means and sign of the effect match the statistical conclusion?
Answer: Yes. Emotional standardized change mean is higher than analytical standardized change mean, and both paired tests are significant.
5. Prompt to self: Does this agree with the directional expectation of Hypothesis 3?
Answer: No. The observed significant difference contradicts the "no significant difference" claim.

\begin{enumerate}
\item A paired-samples $t$-test was conducted to determine if there was a significant difference between standardized Emotional Trust Change and standardized Analytical Trust Change within the WEIRD-like group. Results indicated that Emotional Trust Change ($M = 0.37$, $SD = 1.08$) was significantly higher than Analytical Trust Change ($M = -0.03$, $SD = 0.90$), $t(119) = 3.30$, $p = .001$, $d = 0.30$.
\item A Wilcoxon signed-rank robustness test also indicated a significant within-WEIRD difference, $W = 2436.0$, $p = .002$.
\end{enumerate}

Decision for Hypothesis 3: Supported.

Hypothesis 4: The NORMAL (Non-WEIRD) group will demonstrate a significant difference in the change of pre-post trust scores when comparing emotional versus analytical trust.

Evaluation checklist (Hypothesis 4):
1. Prompt to self: Are the compared variables scale-aligned?
Answer: Yes, by using z-scored emotional and analytical change values (same approach as Hypothesis 3).
2. Prompt to self: Is the chosen inferential test relevant?
Answer: Yes. Paired-samples $t$-test is appropriate for within-participant trust-type difference.
3. Prompt to self: Is there a robustness test?
Answer: Yes. Wilcoxon signed-rank was run in parallel.
4. Prompt to self: Do parametric and non-parametric results agree?
Answer: No. The paired $t$-test is marginal/non-significant at $\alpha = .05$ while Wilcoxon is significant.
5. Prompt to self: What is the sensible interpretation?
Answer: Evidence is mixed, with small effect size, so support is weak/partial rather than definitive.

\begin{enumerate}
\item A paired-samples $t$-test was conducted to determine if there was a significant difference between standardized Emotional Trust Change and standardized Analytical Trust Change within the Non-WEIRD-like group. Results indicated that Emotional Trust Change ($M = -0.11$, $SD = 0.95$) was not significantly different from Analytical Trust Change ($M = 0.01$, $SD = 1.03$), $t(383) = -1.92$, $p = .055$, $d = -0.10$.
\item A Wilcoxon signed-rank robustness test suggested a small but statistically detectable median difference, $W = 32037.0$, $p = .024$.
\end{enumerate}

Decision for Hypothesis 4: Mixed evidence (partial support only under non-parametric robustness testing).

Hypothesis 5: There will be a significant difference in the effect that condition has on trust change size between the NORMAL (Non-WEIRD) and WEIRD subgroups.

Evaluation checklist (Hypothesis 5):
1. Prompt to self: Is this an interaction hypothesis?
Answer: Yes. It asks whether the condition effect differs between subgroups, which is a condition-by-group interaction.
2. Prompt to self: Is the primary test relevant?
Answer: Yes. An OLS interaction model on overall trust change directly tests the $Condition \times WEIRD$ term.
3. Prompt to self: Is there a robustness check for model assumptions?
Answer: Yes. A permutation Difference-in-Differences (DID) test was added.
4. Prompt to self: Do subgroup means align with inferential findings?
Answer: Yes. Group-condition means differ somewhat, but the interaction estimate is small relative to uncertainty.
5. Prompt to self: Is the final conclusion stable across both inferential approaches?
Answer: Yes. Both OLS interaction and permutation DID are non-significant.

\begin{enumerate}
\item A linear interaction model was conducted to test whether condition had a different effect on overall trust change for WEIRD-like vs Non-WEIRD-like participants. The interaction term ($Condition \times WEIRD$) was $b = -0.96$ ($SE = 1.65$), $t(500) = -0.58$, $p = .560$, indicating no statistically significant condition-by-group interaction.
\item A permutation DID robustness test yielded observed $DID = -0.96$ with $p = .553$, consistent with the non-significant interaction result.
\end{enumerate}

Decision for Hypothesis 5: Not supported.

Hypothesis 5a (additional analysis): The effect of condition on overall trust change will be tested independently within WEIRD-like and Non-WEIRD-like subgroups.

Evaluation checklist (Hypothesis 5a):
1. Prompt to self: Is this analysis aligned with the requested subgroup-independent condition tests?
Answer: Yes. It estimates Interactive vs Text effects separately inside WEIRD-like and Non-WEIRD-like groups.
2. Prompt to self: Is the inferential method appropriate for within-subgroup condition comparisons?
Answer: Yes. Welch's independent-samples $t$-tests were used within each subgroup, with Mann-Whitney U robustness checks.
3. Prompt to self: Is there a significant condition effect in the WEIRD-like subgroup alone?
Answer: No. The Interactive vs Text comparison is not significant in WEIRD-like participants.
4. Prompt to self: Is there a significant condition effect in the Non-WEIRD-like subgroup alone?
Answer: No. The Interactive vs Text comparison is not significant in Non-WEIRD-like participants.
5. Prompt to self: Do subgroup-specific findings remain consistent with Hypothesis 5's interaction-level conclusion?
Answer: Yes. Both subgroup-specific tests are non-significant, consistent with the non-significant interaction in Hypothesis 5.

\begin{enumerate}
\item A Welch's independent-samples $t$-test was conducted within the WEIRD-like subgroup to determine if Overall Trust Change differed between Interactive and Text conditions. Results indicated that Interactive ($M = -2.73$, $SD = 8.25$) was not significantly different from Text ($M = -1.97$, $SD = 6.68$), $t(113.15) = -0.56$, $p = .577$, $d = -0.10$.
\item A Welch's independent-samples $t$-test was conducted within the Non-WEIRD-like subgroup to determine if Overall Trust Change differed between Interactive and Text conditions. Results indicated that Interactive ($M = -3.98$, $SD = 8.19$) was not significantly different from Text ($M = -4.18$, $SD = 7.85$), $t(381.31) = 0.24$, $p = .809$, $d = 0.02$.
\item Mann-Whitney U robustness checks were also non-significant in both subgroups (WEIRD-like $p = .604$; Non-WEIRD-like $p = .700$).
\end{enumerate}

Decision for Hypothesis 5a: Not supported.

Hypothesis 5b (additional analysis): Within each condition independently, WEIRD-like and Non-WEIRD-like participants will differ in overall trust change.

Evaluation checklist (Hypothesis 5b):
1. Prompt to self: Is this analysis a meaningful complement to Hypotheses 5 and 5a?
Answer: Yes. It tests subgroup differences inside each condition, complementing the condition differences tested inside each subgroup in 5a.
2. Prompt to self: Is the inferential method appropriate for within-condition subgroup comparisons?
Answer: Yes. Welch's independent-samples $t$-tests were run within Interactive and within Text, with Mann-Whitney U robustness checks.
3. Prompt to self: Is there evidence of a subgroup difference within the Interactive condition?
Answer: No. WEIRD-like and Non-WEIRD-like change scores are not significantly different in Interactive.
4. Prompt to self: Is there evidence of a subgroup difference within the Text condition?
Answer: Yes under Welch's $t$-test, with WEIRD-like participants showing less negative change than Non-WEIRD-like participants; the Mann-Whitney robustness check is not significant.
5. Prompt to self: What is the most defensible interpretation?
Answer: Evidence is condition-specific and modest: no subgroup difference in Interactive, and mixed/partial subgroup difference in Text.

\begin{enumerate}
\item A Welch's independent-samples $t$-test was conducted within the Interactive condition to determine if Overall Trust Change differed between WEIRD-like and Non-WEIRD-like participants. Results indicated that WEIRD-like scores ($M = -2.73$, $SD = 8.25$) were not significantly different from Non-WEIRD-like scores ($M = -3.98$, $SD = 8.19$), $t(98.15) = 1.02$, $p = .309$, $d = 0.15$.
\item A Welch's independent-samples $t$-test was conducted within the Text condition to determine if Overall Trust Change differed between WEIRD-like and Non-WEIRD-like participants. Results indicated that WEIRD-like scores ($M = -1.97$, $SD = 6.68$) were significantly higher than Non-WEIRD-like scores ($M = -4.18$, $SD = 7.85$), $t(114.29) = 2.14$, $p = .034$, $d = 0.29$.
\item Mann-Whitney U robustness checks were non-significant in both conditions (Interactive $p = .431$; Text $p = .076$).
\end{enumerate}

Decision for Hypothesis 5b: Mixed evidence (partial support driven by the Text-condition Welch test only).
