# Elliott Wave Knowledge Base

## Purpose

This repository contains a structured Elliott Wave knowledge base designed for:

* Transformer training
* RAG (Retrieval-Augmented Generation)
* Trading transcript analysis
* Pattern classification
* Forecast generation
* Elliott Wave reasoning

The architecture intentionally separates different types of knowledge into independent domains.

---

# Separation Philosophy

The repository follows a strict principle:

> One file = One responsibility

A common mistake is storing pattern definitions, trader terminology, examples, and forecasting rules in a single file.

While convenient for humans, this creates ambiguity for machine learning systems.

Instead, this repository separates knowledge into independent layers:

1. Pattern Definitions
2. Trader Language
3. Classification Logic
4. Fibonacci Logic
5. Historical Examples
6. Market Context

Each layer answers a different question.

---

## Why Separation Matters

Consider the statement:

> "This looks like a liquidity grab before a major decline."

This sentence contains:

* Trader vocabulary
* Market context
* Pattern implication

It does NOT define an Elliott pattern.

Therefore it should not exist inside pattern definitions.

Likewise:

> "Expanded Flat is a 3-3-5 correction."

This is a canonical pattern definition.

It should not exist inside trader vocabulary.

Separating domains reduces ambiguity and improves retrieval quality.

---

# Repository Structure

```text
knowledge/

├── elliott_patterns.json
├── elliott_aliases.json
├── elliott_classifier_rules.json
├── elliott_fibonacci_rules.json
├── elliott_examples.json
└── elliott_market_context.json
```

---

# File Responsibilities

## 1. elliott_patterns.json

### Purpose

Stores canonical Elliott Wave definitions.

This file represents the ground truth.

### Contains

* Zigzag
* Double Zigzag
* Triple Zigzag
* Regular Flat
* Expanded Flat
* Running Flat
* Contracting Triangle
* Barrier Triangle
* Expanding Triangle
* Running Triangle
* Double Three
* Triple Three

### Example

```json
{
  "id": "expanded_flat",
  "structure": "3-3-5",
  "rules": [
    "B_exceeds_A_start",
    "C_extends_beyond_A_end"
  ]
}
```

### Must NOT Contain

* Liquidity grab
* Bull trap
* Bear trap
* Examples
* Forecasts
* Fibonacci targets

Reason:

This file defines what a pattern is.

Nothing else.

---

## 2. elliott_aliases.json

### Purpose

Maps trader language to canonical patterns.

This file acts as a translator between market participants and Elliott terminology.

### Example

```json
{
  "term": "Non Ideal ABC",
  "possible_patterns": [
    "expanded_flat"
  ]
}
```

### Contains

* ABC Correction
* Non-Ideal ABC
* Complex Correction
* Running Correction
* Deep Correction

### Must NOT Contain

* Elliott rules
* Fibonacci targets
* Classification logic

Reason:

This file explains how traders speak.

Not how patterns work.

---

## 3. elliott_classifier_rules.json

### Purpose

Stores reasoning logic used for pattern identification.

### Example

```json
{
  "if": "A=3_and_B=3_and_C=5",
  "classify_as": "flat_family"
}
```

### Contains

* Wave count rules
* Candidate selection logic
* Validation rules
* Confidence scoring logic

### Must NOT Contain

* Pattern descriptions
* Community terminology
* Examples

Reason:

This file explains how to identify a pattern.

Not what the pattern means.

---

## 4. elliott_fibonacci_rules.json

### Purpose

Stores projection and validation relationships.

### Example

```json
{
  "pattern": "expanded_flat",
  "common_targets": [
    "1.236x_A",
    "1.618x_A"
  ]
}
```

### Contains

* Retracements
* Extensions
* Target calculations
* Projection formulas

### Must NOT Contain

* Pattern definitions
* Community terminology

Reason:

Price projection is a separate domain from pattern classification.

---

## 5. elliott_examples.json

### Purpose

Stores annotated historical examples.

### Example

```json
{
  "symbol": "BTCUSDT",
  "timeframe": "4H",
  "pattern": "expanded_flat",
  "outcome": "bearish_reversal"
}
```

### Contains

* Historical charts
* Labels
* Outcomes
* Validation results

### Must NOT Contain

* Classification rules
* Pattern definitions

Reason:

Examples are evidence.

Rules are knowledge.

They should not be mixed.

---

## 6. elliott_market_context.json

### Purpose

Stores modern trading concepts frequently used alongside Elliott Wave.

### Contains

* Liquidity Grab
* Stop Hunt
* Bull Trap
* Bear Trap
* Break of Structure (BOS)
* Market Structure Shift (MSS)
* Accumulation
* Distribution
* Fair Value Gap (FVG)

### Example

```json
{
  "term": "Liquidity Grab",
  "description": "Temporary move beyond a key level to trigger stops."
}
```

### Must NOT Contain

* Elliott structures
* Wave counts

Reason:

Market concepts are context.

Patterns are structure.

---

# Knowledge Flow

The intended reasoning pipeline is:

```text
Raw Transcript
        ↓
Trader Vocabulary Mapping
        ↓
Market Context Detection
        ↓
Canonical Pattern Mapping
        ↓
Pattern Classification
        ↓
Fibonacci Validation
        ↓
Confidence Scoring
        ↓
Forecast Generation
```

---

# Example Workflow

Input:

"ABC correction with a liquidity sweep above the highs before dropping."

Step 1:

Alias Mapping

```text
ABC Correction
→ Zigzag or Flat candidate
```

Step 2:

Market Context

```text
Liquidity Sweep
→ Possible Expanded Flat
```

Step 3:

Classifier

```text
3-3-5
B exceeds A start
```

Result:

```text
Expanded Flat
```

Step 4:

Fibonacci Validation

```text
C target = 1.236–1.618 × A
```

Final Output:

```text
Pattern: Expanded Flat
Confidence: High
Bias: Bearish
```

---

# Design Goal

The goal is not merely to store Elliott Wave information.

The goal is to create a knowledge architecture where:

* Definitions remain objective.
* Trader language remains flexible.
* Rules remain explainable.
* Forecasting remains auditable.
* Retrieval remains efficient.

This separation allows transformers and RAG systems to reason more accurately than a single monolithic Elliott Wave dataset.
