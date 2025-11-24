2. Session Search
* **What the Code Does:** Scans the first 100 mouse sessions in the database. Adds the first 5 sessions containing neural data (`spikes.times`) to the list.
* **What it finds:** It filters out empty or faulty records. It finds a valid, complete data set (Session ID - `eid`) that can be analysed.

3. Data Loading
* **What the code does:** It downloads two critical files for the selected session:
    1.  `spikes.times`: At what second did the firing occur?
2.  `spikes.clusters`: Which neuron did this firing come from?
* **What it finds:** Loads the raw neural data into the computer's memory. (E.g., ‘There are a total of 2 million spikes.’)

4. Quality Control
* **What the Code Does:** It counts how many times each neuron fired in total. If a neuron fired less than 1000 times, it is considered ‘noise’ or a ‘dead neuron’ and is excluded from the analysis.
* **What it finds:** A list of **‘Good Neurons’**. To ensure the analysis is reliable, it selects only highly active and healthy neurons.

5. Single Neuron Tau Calculation - Test
* **What the code does:** This is the most critical mathematical part of the code. It calculates the **‘Autocorrelation’** for a selected test neuron.
* *Logic:* ‘If this neuron fired just now, what is the probability it will fire again in 100ms?’
* *Tau ($\tau$):* This is the decay time of that probability.
* **What it finds:** It finds the neuron's **‘Memory Duration’.**
    * Short $\tau$ (<50ms): Sensory neuron (Performs instantaneous tasks, forgets).
    * Long $\tau$ (>300ms): Cognitive neuron (Stores information in memory).

6. Brain Regions Mapping
* **What the Code Does:** It downloads the `channels` data and uses the **Allen Brain Atlas** to find the exact location (coordinates) of the neurons in the brain.
* **What it finds:** It finds which neuron is located in which neighbourhood. (E.g.: ‘Neuron #45 is located in the PO region of the Thalamus.’)

7. Region-Based Analysis
* **What the Code Does:** It groups neurons in specified regions (e.g., PO, LP, CA1) and calculates the $\tau$ (memory duration) for each one individually. It then takes the median of these durations.
* **What it Finds:** It finds the **hierarchy of brain regions**.
    * For example: ‘The thalamus (PO) is very fast (50ms), but the hippocampus (CA1) is slow and memory-intensive (200ms).’

8. Circuit-Level Analysis ( The Most Important Part)
* **What the Code Does:** This is the heart of the project. Instead of looking at a single neuron, it looks at **neuron pairs**. It measures the time it takes for two neurons to trigger each other (Cross-Correlation).
    * Formula: `Integration Index = (Pairwise Tau - Single Tau) / Single Tau`
* **What it finds:** It finds out whether that region is a **‘Team Player’**.
    * If `Integration Index > 0`: The neurons combine and hold information longer than they could on their own. This indicates that the region is working for decision-making or memory.

9. Extended & Loose Analysis
* **What the Code Does:** If the filters are too strict in the first analysis, there may not be enough data. Here, it loosens the filters a bit (including neurons that spike less) and spreads the analysis to more regions (SSp, VISp, etc.).
* **What it finds:** It produces a statistically more reliable, broad-based ‘Brain Map’.

10. Statistical Tests
* **What the code does:** It tests whether the differences it finds are due to chance (Mann-Whitney U test, Kruskal-Wallis test).
* **What it finds:** Scientific evidence. It allows us to say ‘The thalamus is significantly slower than the visual cortex’ (p-value < 0.05).

11. Visualisation
* **What the code does:** It draws 3 different graphs.
    1.  **Bar Chart:** Columns showing how fast each region is.
2.  **Violin Plot:** Distribution of neuron speeds.
3.  **Circuit Scatter:** Comparison of single neuron speed vs. circuit speed.
* **What It Finds:** Converts the analysis results into visual evidence that can be used in presentations.
