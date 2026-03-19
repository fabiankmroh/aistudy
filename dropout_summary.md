# Dropout: A Complete Summary

## What Is Dropout?

Dropout is a **regularization technique** designed to prevent overfitting in neural networks. During training, it randomly "kills" (sets to zero) a fraction of neuron activations in each forward pass, forcing the network to learn redundant representations rather than relying on any single neuron.

The key hyperparameter is **p**, the probability that any given neuron's output is set to zero.

---

## Why Dropout Works

Without dropout, neurons can **co-adapt** — they learn to depend on specific other neurons being present. This leads to memorization of the training data. Dropout breaks these co-dependencies by ensuring that no neuron can rely on any other neuron always being there.

Intuitively, dropout trains an **ensemble of subnetworks**. Each forward pass uses a different random subset of neurons, so the model effectively learns many overlapping architectures that share weights. At test time, the full network acts as an approximate average of all these subnetworks.

---

## Critical Clarification: Activations vs. Weights

Before diving into the math, this distinction is essential.

Dropout does **not** modify weights. Weights live on the connections between neurons and are only updated by the optimizer during backpropagation.

What dropout modifies are **activations** — the output values that neurons produce after applying their weights and activation function. These are the values flowing through the network during the forward pass.

| Concept | What It Is | Modified by Dropout? |
|---|---|---|
| **Weights** | Parameters on connections between neurons | **No** |
| **Activations** | Output values produced by neurons | **Yes** (zeroed or scaled) |

When we say "a neuron's output gets multiplied by 2," we mean the activation value flowing forward through the network — the weights on that neuron's incoming or outgoing connections remain untouched.

---

## Standard (Vanilla) Dropout

The textbook version works as follows:

**During training**, for each neuron with activation *a*:
- With probability *p*: output = 0
- With probability (1 − p): output = *a*

The expected output of a single neuron during training is:

```
E[output] = p × 0 + (1 − p) × a = (1 − p) × a
```

This is **less than** the original activation *a* by a factor of (1 − p).

**During testing**, to compensate:
- Multiply all activations by (1 − p)

This ensures the expected magnitude at test time matches training time.

### The Problem with Vanilla Dropout

1. Your inference code must know the value of *p* used during training.
2. If you change *p* during training (e.g., for experimentation), you must update inference accordingly.
3. Comparing models trained with different *p* values becomes messy — each needs different test-time scaling.

---

## Inverted Dropout: The Full Picture

### The Core Idea

Inverted dropout flips the compensation from test time to train time. Instead of scaling down at inference, you scale up the surviving neurons during training so their expected value already matches the no-dropout case. At test time, you do absolutely nothing.

This is what **PyTorch, TensorFlow, and all modern frameworks actually implement** when you use `nn.Dropout`.

### Mathematical Formulation

Let *a* be a neuron's activation before dropout, and *p* be the drop probability.

Define a random mask variable *m* for each neuron:

```
m ~ Bernoulli(1 − p)

    m = 0  with probability p       (neuron is dropped)
    m = 1  with probability (1 − p) (neuron survives)
```

**Vanilla dropout** computes the output as:

```
output_train = m × a
output_test  = (1 − p) × a
```

**Inverted dropout** computes the output as:

```
output_train = (m × a) / (1 − p)
output_test  = a
```

The division by (1 − p) during training is the key difference.

### Proof That Expected Values Are Preserved

For inverted dropout, the expected output during training:

```
E[output_train] = E[ (m × a) / (1 − p) ]
                = a / (1 − p) × E[m]
                = a / (1 − p) × (1 − p)
                = a
```

The expected output during testing:

```
E[output_test] = a
```

**They match.** The expected activation is *a* in both phases — no correction needed anywhere.

For comparison, vanilla dropout during training:

```
E[output_train] = E[m × a] = a × (1 − p)
```

This equals *a* only after you remember to multiply by (1 − p) at test time. If you forget, your network's outputs are systematically too large by a factor of 1/(1 − p).

### What the Scaling Factor Actually Does

The factor **1/(1 − p)** compensates for the "missing" neurons. Here's the intuition:

If you drop 50% of neurons (p = 0.5), only half are active. Each survivor must carry **twice** the signal to maintain the same total: 1/(1 − 0.5) = 2.

If you drop 20% (p = 0.2), 80% survive. Each must carry 1/0.8 = **1.25×** the signal.

If you drop 80% (p = 0.8), only 20% survive. Each must carry 1/0.2 = **5×** the signal.

The more aggressive the dropout, the larger the compensation factor for survivors.

### Worked Example: Single Neuron at Various p Values

A single neuron outputs activation = 10. Here is what inverted dropout does at different drop rates:

**p = 0.2 (drop 20%):**

```
With prob 0.2: output = 0
With prob 0.8: output = 10 / (1 − 0.2) = 10 / 0.8 = 12.5

E[output] = 0.2 × 0 + 0.8 × 12.5 = 0 + 10 = 10 ✓
```

**p = 0.5 (drop 50%):**

```
With prob 0.5: output = 0
With prob 0.5: output = 10 / (1 − 0.5) = 10 / 0.5 = 20

E[output] = 0.5 × 0 + 0.5 × 20 = 0 + 10 = 10 ✓
```

**p = 0.8 (drop 80%):**

```
With prob 0.8: output = 0
With prob 0.2: output = 10 / (1 − 0.8) = 10 / 0.2 = 50

E[output] = 0.8 × 0 + 0.2 × 50 = 0 + 10 = 10 ✓
```

Regardless of *p*, the expected value is always preserved at 10. The test-time output is also 10 with no scaling — everything is consistent.

### Worked Example: Layer-Level View (Why Scaling Matters)

This example shows *why* the scaling is necessary, not just that it works.

Consider a layer with 4 neurons, each outputting activation value 10. The next layer expects a total input of **40**.

**Without dropout:**

```
Neuron 1: 10    Neuron 2: 10    Neuron 3: 10    Neuron 4: 10
Total to next layer: 40
```

**Naive dropout (p = 0.5, no scaling) — suppose neurons 2 and 3 are dropped:**

```
Neuron 1: 10    Neuron 2: 0     Neuron 3: 0     Neuron 4: 10
Total to next layer: 20  ← HALF of what the next layer expects!
```

The next layer was trained to work with inputs around 40. Receiving 20 instead will produce distorted outputs, wrong gradients, and unstable training. This is the fundamental problem that scaling solves.

**Inverted dropout (p = 0.5) — same neurons 2 and 3 dropped:**

```
Neuron 1: 10/0.5 = 20    Neuron 2: 0    Neuron 3: 0    Neuron 4: 10/0.5 = 20
Total to next layer: 40  ← matches the no-dropout case!
```

At test time: all 4 neurons output 10, total = 40. No scaling needed.

**Inverted dropout (p = 0.3) — now with 10 neurons, each outputting 10:**

On average, 7 neurons survive. Without scaling, the total would be 70 out of an expected 100.

```
Each survivor: 10 / (1 − 0.3) = 10 / 0.7 ≈ 14.3
Expected total: 7 × 14.3 ≈ 100 ✓
```

**Inverted dropout (p = 0.8) — very aggressive, 10 neurons, each outputting 10:**

On average, only 2 neurons survive. Without scaling, the total would be 20 out of an expected 100.

```
Each survivor: 10 / (1 − 0.8) = 10 / 0.2 = 50
Expected total: 2 × 50 = 100 ✓
```

No matter how aggressive the dropout, the expected total input to the next layer is always preserved.

### Summary: Vanilla vs. Inverted Side by Side

| Aspect | Vanilla Dropout | Inverted Dropout |
|---|---|---|
| Training: dropped neurons | Output = 0 | Output = 0 |
| Training: surviving neurons | Output = *a* (unchanged) | Output = *a* / (1 − p) (scaled up) |
| Training: E[output] | (1 − p) × *a* (reduced) | *a* (preserved) |
| Testing | Multiply all by (1 − p) | **Do nothing** |
| Testing: E[output] | *a* (after scaling) | *a* (naturally) |

### Why Inverted Dropout Is Superior in Practice

**1. Decouples training from inference.**
Your test-time forward pass is the plain network — no dropout logic, no scaling, no knowledge of *p*. You can export the model and deploy it without carrying any dropout metadata.

**2. Enables dropout rate experimentation.**
You can change *p* mid-training (e.g., curriculum dropout, where you increase *p* as training progresses) without worrying about how it affects inference. Every model, regardless of what *p* was used, produces correctly scaled outputs at test time.

**3. Simplifies model comparison.**
Two models trained with p = 0.3 and p = 0.7 both produce outputs at the same scale during evaluation. You can directly compare their test accuracies without wondering if you applied the right scaling factor.

**4. Cleaner gradient flow during training.**
Because the surviving activations are scaled up, the gradients flowing backward through those neurons are also scaled by 1/(1 − p). This means the effective learning rate for surviving neurons is amplified, which partially compensates for the fact that each neuron only receives gradients on a fraction of batches.

**5. No "forgetting to scale" bugs.**
With vanilla dropout, forgetting the test-time scaling is a silent bug — the network still produces outputs, they're just wrong in magnitude. Inverted dropout eliminates this failure mode entirely.

---

## Dropout in PyTorch

### Basic Usage

```python
import torch.nn as nn

# In model definition
self.dropout = nn.Dropout(p=0.3)  # 30% drop rate

# In forward pass
x = self.dropout(x)
```

PyTorch's `nn.Dropout(p)` implements **inverted dropout**:
- During `model.train()`: zeros activations with probability *p*, divides survivors by (1 − p).
- During `model.eval()`: complete no-op, passes input through unchanged.

### Train vs. Eval Mode

```python
model.train()  # Dropout is ACTIVE (zeros + scales)
model.eval()   # Dropout is a NO-OP (passes input through unchanged)
```

**Always call `model.eval()` before inference and `model.train()` before training.**

### Spatial Dropout for CNNs

For convolutional networks, `nn.Dropout2d` drops entire feature maps (channels) instead of individual pixels:

```python
self.spatial_dropout = nn.Dropout2d(p=0.1)  # After conv/pool layers
```

This is more effective for CNNs because adjacent pixels in a feature map are highly correlated — dropping individual pixels has little effect since neighbors carry similar information.

---

## Where to Place Dropout in a Network

A typical pattern for a CNN classifier:

```python
self.classifier = nn.Sequential(
    nn.Dropout(0.3),              # Before first FC layer
    nn.Linear(128 * 4 * 4, 512),
    nn.ReLU(),
    nn.Dropout(0.3),              # Before output layer
    nn.Linear(512, 10)
)
```

General guidelines:
- **After activation functions** (ReLU, etc.) in fully connected layers.
- **After pooling layers** in CNNs (using `Dropout2d`).
- **Not typically after the final output layer** — you want the full output for the loss function.
- **Not usually between conv layers** within the same block — use `Dropout2d` sparingly here.

---

## Common p Values

| Context | Typical p | Scaling Factor 1/(1−p) |
|---|---|---|
| Fully connected layers | 0.3 – 0.5 | 1.43× – 2.0× |
| Convolutional layers (Dropout2d) | 0.05 – 0.2 | 1.05× – 1.25× |
| After embeddings (NLP) | 0.1 – 0.3 | 1.11× – 1.43× |
| Small datasets | 0.5+ | 2.0×+ |
| Large datasets | 0.1 – 0.3 | 1.11× – 1.43× |

Note how more aggressive dropout requires larger compensation — dropping 80% of neurons means each survivor carries 5× the signal, which introduces high variance. This is why very high *p* values are rarely used in practice.

---

## Dropout vs. Other Regularization

| Technique | Mechanism | When to Use |
|---|---|---|
| **Dropout** | Randomly zeros activations during training | Large FC layers, overfitting gap |
| **Weight Decay (L2)** | Penalizes large weights via optimizer | General-purpose, always a good default |
| **Data Augmentation** | Artificially expands training set | Image tasks, often the biggest single gain |
| **Early Stopping** | Stop training when validation loss plateaus | Simplest approach, no code changes |
| **Batch Normalization** | Normalizes layer inputs, adds slight noise | Already provides mild regularization |

These techniques are **complementary** — in practice, you often combine several of them.

---

## Key Takeaways

1. **Dropout zeros activations, not weights.**
2. **Inverted dropout** scales survivors by 1/(1−p) during training so that `E[output] = a` in both training and testing — no correction needed at inference.
3. The mathematical proof is straightforward: `E[(m × a)/(1−p)] = a/(1−p) × (1−p) = a`.
4. The scaling factor 1/(1−p) compensates for missing neurons: 50% dropped → 2× scaling, 20% dropped → 1.25× scaling, 80% dropped → 5× scaling.
5. In PyTorch, `nn.Dropout(p)` implements inverted dropout — just toggle between `model.train()` and `model.eval()`.
6. Inverted dropout's main advantages are: decoupled train/test logic, safe hyperparameter experimentation, cleaner gradient flow, and elimination of scaling bugs.
7. If you haven't learned regularization techniques yet, **reducing epochs (early stopping)** is the simplest way to combat overfitting.
