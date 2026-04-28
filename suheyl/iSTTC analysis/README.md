<img width="792" height="330" alt="Ekran görüntüsü 2026-02-03 183257" src="https://github.com/user-attachments/assets/d9c2eaa3-cc6b-4323-80b0-c0a58dda962d" />

for this graphs:
This visualization shows the **intrinsic timescale analysis** of a single neuron (Cluster 0.0) using the iSTTC (inter-Spike-Time Tiling Coefficient) method:

## **Left Panel: Spike Raster (first 1000 spikes)**
- Shows the temporal pattern of the first 1000 action potentials (spikes)
- X-axis: Time in seconds
- Each vertical blue line represents one spike
- The neuron fires regularly and consistently at ~15 Hz
- The relatively uniform spacing indicates stable, tonic firing pattern

## **Right Panel: Intrinsic Timescale**
- **Blue dots (iSTTC)**: The inter-spike-time tiling coefficient calculated at different time lags
- **Red curve (Fit)**: Exponential decay function fitted to the data
- **τ = 50.6 ms**: The intrinsic timescale - the "temporal memory" of this neuron

## **What does τ = 50.6 ms mean?**

The intrinsic timescale represents **how long the neuron "remembers" its past activity**:

- **Short timescale (~50ms)**: Indicates **fast dynamics**
  - The neuron responds quickly to inputs
  - Past activity influences current firing for only ~50 milliseconds
  - Typical of neurons involved in **sensory processing** or **rapid motor responses**
  - Suggests the neuron integrates information over short time windows

- If it were **long timescale (>200ms)**: Would indicate **slow dynamics**
  - Associated with **decision-making**, **working memory**, or **sustained attention**
  - Neurons would integrate information over longer periods

## **The decay curve interpretation:**
- **Steep initial drop**: Strong temporal correlation at short lags (neurons tend to fire in bursts or regular patterns)
- **Rapid approach to baseline**: The temporal influence decays quickly
- **Flat tail**: Beyond ~200ms, there's no significant temporal correlation - past activity no longer influences current firing

This analysis is part of understanding the **temporal computational properties** of individual neurons in neural circuits.


<img width="808" height="698" alt="Ekran görüntüsü 2026-02-03 204241" src="https://github.com/user-attachments/assets/c428371b-9578-4a7c-8b67-48d1b95206a4" />

## Overall purpose of the figure

This figure compares **single-timescale** versus **multi-timescale** models of neuronal firing and analyzes the **temporal structure of neural activity** in terms of fast and slow intrinsic timescales (τ). The goal is to test whether adding multiple timescales meaningfully improves model fit, and how these timescales relate to firing rate and session variability.

## 1️⃣ Fit Quality: Single vs Multi (top-left)

* **X-axis:** R² of the single-timescale model
* **Y-axis:** R² of the multi-timescale model
* **Red dashed line:** identity line (equal performance)

**Interpretation:**
Most points lie very close to the identity line, indicating that the multi-timescale model usually provides only a **slight improvement** over the single-timescale model. Large gains are rare.

## 2️⃣ Multi-fit Improvement (ΔR² = Multi − Single) (top-middle)

* Histogram of ΔR² values
* **Mean ΔR² ≈ 0.0115**

**Interpretation:**
The distribution is sharply centered near zero, showing that:

* Improvements from adding extra timescales are **small but systematic**
* The multi-timescale model rarely overfits or dramatically outperforms the single model

## 3️⃣ Multi-Timescale Necessity (top-right pie chart)

* **~92.2%:** single-timescale sufficient
* **~7.8%:** multi-timescale necessary (ΔR² > threshold)

**Interpretation:**
Only a small subset of neurons truly require multiple intrinsic timescales to be well described. For the vast majority, a single dominant timescale captures the temporal dynamics adequately.

## 4️⃣ Fast vs Slow Timescales (middle-left)

* **X-axis:** fast timescale τ_fast (ms)
* **Y-axis:** slow timescale τ_slow (ms)
* Log-scaled axes

**Interpretation:**
Fast and slow timescales are clearly separated:

* τ_fast typically lies in the **tens of milliseconds**
* τ_slow spans **hundreds of milliseconds to seconds**

This supports the existence of **hierarchically distinct temporal processes** within neurons.

## 5️⃣ Timescale Distributions (middle-center)

* Histogram of τ values:

  * Single-timescale τ
  * Fast τ
  * Slow τ

**Interpretation:**
The single-timescale distribution overlaps strongly with the **fast timescale**, suggesting that when only one τ is fitted, it primarily captures **fast dynamics**, while slower components are missed.

## 6️⃣ Component Weights (middle-right)

* **X-axis:** fast component weight
* **Y-axis:** slow component weight

**Interpretation:**
Fast components usually dominate the variance explained. Slow components contribute less but are non-negligible in a subset of neurons, consistent with the earlier “multi-timescale necessity” result.

## 7️⃣ Firing Rate vs Fast Timescale (bottom-left)

* **X-axis:** firing rate (Hz, log scale)
* **Y-axis:** τ_fast (ms)

**Interpretation:**
Higher firing rates tend to be associated with **shorter fast timescales**, consistent with the idea that rapidly firing neurons integrate information over shorter temporal windows.

## 8️⃣ Firing Rate vs Slow Timescale (bottom-middle)

* **X-axis:** firing rate (Hz, log scale)
* **Y-axis:** τ_slow (ms)

**Interpretation:**
The slow timescale shows a much weaker relationship with firing rate, suggesting that slow temporal integration is governed by mechanisms largely independent of instantaneous spiking activity.

## 9️⃣ Timescale Separation (bottom-right)

* Histogram of τ_slow / τ_fast ratios
* **Median separation ≈ 5.8×**

**Interpretation:**
Fast and slow processes are typically separated by almost **an order of magnitude**, reinforcing the idea that they reflect **distinct biological or circuit-level mechanisms** rather than fitting artifacts.

## 🔢 Session-wise Summary Table (bottom)

The table reports statistics for each recording session:

* Mean, median, and standard deviation of single-timescale τ
* Mean τ_fast and τ_slow
* Number of neurons per session

**Interpretation:**
Timescale statistics are **consistent across sessions**, indicating that the observed temporal structure is stable and not driven by session-specific artifacts.

## 🔑 Key takeaway

> Most neurons are well described by a **single intrinsic timescale**, but a **small, systematic subset** exhibits genuine multi-timescale dynamics, with fast and slow processes separated by nearly an order of magnitude. Single-timescale models primarily capture fast dynamics, while multi-timescale models reveal additional slow integration components.

<img width="811" height="580" alt="Ekran görüntüsü 2026-02-03 204258" src="https://github.com/user-attachments/assets/1fecfcf1-a72c-4eb4-b6d3-d8bc7fd1a620" />

Below is a **clear, panel-by-panel explanation in English** of this second figure and the table at the top. I’ll also connect it explicitly to the **single vs multi-timescale question**, so the interpretation is coherent with the previous figure.

---

## Overall purpose of this figure

This figure examines **session-to-session consistency** in:

* intrinsic timescales (fast and slow),
* firing rates,
* model-fit improvement (ΔR²),
* and neuron counts,

to test whether the results are **stable across recording sessions** or driven by a particular dataset.

## 🔢 Session summary table (top)

| Session | Mean ΔR² | Mean firing rate (Hz) |
| ------- | -------- | --------------------- |
| 1       | 0.01     | 5.83                  |
| 2       | 0.01     | 5.95                  |
| 3       | 0.01     | 7.16                  |
| 4       | 0.01     | 6.64                  |
| 5       | 0.01     | 8.18                  |
| 6       | 0.02     | 7.13                  |

### Interpretation

* **ΔR² is consistently ~0.01 across all sessions**, indicating that:

  * the small advantage of the multi-timescale model is **not session-specific**
* Mean firing rates vary moderately but remain within a similar range (≈ 6–8 Hz)

This suggests that **timescale effects are not driven by firing-rate differences across sessions**.

## 1️⃣ Slow Timescale Across Sessions (top-left)

* **Y-axis:** τ_slow (ms, very large range)
* **X-axis:** session number

### Interpretation

* Large variability in τ_slow values **within each session**
* Similar distributions across sessions
* No obvious session with systematically longer or shorter slow timescales

➡️ Slow timescales are **heterogeneous at the neuron level**, but **stable at the session level**.

## 2️⃣ Fast Timescale Across Sessions (top-middle)

* Boxplots of τ_fast per session

### Interpretation

* Median τ_fast is similar across all sessions
* Slight session-to-session variation, but strong overlap

➡️ Fast timescales are **highly consistent across sessions**, supporting a shared underlying physiological mechanism.

## 3️⃣ Slow Timescale Across Sessions (top-right)

* Boxplots of τ_slow per session

### Interpretation

* τ_slow medians are again similar across sessions
* Distributions are wider than τ_fast

➡️ Slow dynamics are **more variable**, but still not session-dependent.

## 4️⃣ Neurons per Session (bottom-left)

* Bar plot of neuron counts

### Interpretation

* Each session contains hundreds to ~1000 neurons
* Variation in neuron count does **not correlate** with changes in timescales or ΔR²

➡️ Results are not an artifact of unequal sampling.

## 5️⃣ Fit Improvement Across Sessions (bottom-middle)

* **Y-axis:** ΔR² (multi − single)
* Red dashed line at 0

### Interpretation

* ΔR² distributions are centered slightly above zero in all sessions
* Similar spread across sessions

➡️ The small but consistent improvement of the multi-timescale model is **reproducible across sessions**, not driven by a single recording.

## 6️⃣ Firing Rate Across Sessions (bottom-right)

* **Y-axis:** firing rate (Hz, log scale)

### Interpretation

* Firing rate distributions are similar across sessions
* Minor upward shift in some sessions, but no drastic differences

➡️ Confirms that **session differences in firing rate do not explain timescale or fit differences**.

## 🔑 Key takeaway (this figure)

> Intrinsic timescales, firing rates, and multi-timescale model improvements are **remarkably consistent across recording sessions**. The small ΔR² gain from multi-timescale models is reproducible, while both fast and slow timescales show neuron-level heterogeneity but session-level stability.


