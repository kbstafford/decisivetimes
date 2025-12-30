The DSI Distribution chart is the "Confidence Report Card" for the mouse. Since we cannot ask the mouse how "confident" it feels, we calculate it using this weighted formula:DSI = (0.3{Contrast}) + (0.3{RT Score}) + (0.4{Accuracy})
Contrast: The more distinct the stimulus, the higher the confidence.
RT Score: The faster the mouse made a decision (short Reaction Time), the more confident it was.
Accuracy: Correct decisions are generally associated with higher confidence.
Categorization: Trials were divided into “High” and “Low” based on the median DSI value.

Output analysis:
• Firing Rate Comparison: There is only a 1.0% difference between High Confidence (13.68 Hz) and Low Confidence (13.55 Hz). This difference is biologically negligible.
• P-value (1.000000): This indicates that there is no statistically significant difference between these two groups.
• Active Neurons: Contrary to expectations, slightly more neurons were active in the Low Confidence trials (141.4) compared to High Confidence (137.8).
• Spearman ρ (-0.025): Firing rate does not increase as confidence increases; instead, there is a very slight decrease (negative correlation).

Graph Results: 
• DSI Distribution (Histogram):
    • What does it show? How many trials are “strong” decisions and how many are “weak” decisions.
    • Interpretation: The distribution around the median line (red) shows whether the confidence levels of the trials are homogeneously distributed.
• Firing Rate by Confidence (Boxplot):
    • What does it show? The difference in speed between the two groups.
    • Interpretation: The heights and midlines of the boxes (green and red) are very close to each other. This is the main evidence for why the hypothesis was rejected in this data.
• FR Distribution (Violin Plot):
    • What does it show? The density distribution of neuron firing rates.
    • Interpretation: The fact that the shapes (violins) are identical proves that neurons exhibit similar “population behavior” in both cases.
• DSI vs Firing Rate (Scatter Plot):
    • What does it show? The relationship between the individual confidence score in each trial and the firing rate.
    • Interpretation: The points are very scattered, and the red “Fit” line is almost horizontal. This shows that we cannot say “As confidence increases, speed increases.”
• Binned Analysis (Line Plot):
    • What does it show? The average firing rate for each of the 10 bins into which DSI is divided.
    • Interpretation: If the line zigzags or is flat (as in your data), there is no linear relationship between the variables.
• Active Neurons per Trial (Boxplot):
    • What does it show? The difference in the “number of neurons participating in the decision.”
Interpretation: You will see that the Low Confidence side is slightly higher. This can be interpreted as the brain trying to engage more neurons in difficult decisions (low confidence).
