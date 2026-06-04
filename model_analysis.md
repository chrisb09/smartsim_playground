# Transformer Model Inference Analysis

The transformer-based Torch model used in this project (specifically the `transformer_tbl` class and its `TransformerInferenceWrapper`) requires the following inputs for inference:

## 1. Data Inputs (Tensors)
When using the core `transformer_tbl` model's `forward` method, it requires **4 inputs**:
*   **`src` (Source Sequence):** The input history of the flow field.
    *   **Shape:** `(seq_len, batch, feature_dim)` (or `(batch, seq_len, feature_dim)` if `batch_first=True`).
    *   **Content:** Flattened 3D cubes of flow field data (e.g., velocity components).
*   **`tgt` (Target Sequence):** The sequence being predicted. During iterative inference, this starts as the last step of `src` and grows as predictions are appended.
*   **`src_mask` & `tgt_mask`:** Triangular masks used by the transformer to ensure causality.

The **`TransformerInferenceWrapper`** (the TorchScript-friendly version used in production) simplifies this to **1 primary input**:
*   **`src`:** The model handles the generation of `tgt` and masks internally during its forecasting loop.

---

## 2. Input Dimensions and Specifics
Based on the project's configuration:

*   **Sequence Length (`mlInputLength`):** **5** past time steps.
*   **Feature Dimension (`inp_dim`):** **512**. Derived from the "cube" size: $8 \times 8 \times 8 = 512$ (where `mlCubeD = 8`).
*   **Number of Fields:** **3** (presumably U, V, and W velocity components).
*   **Batching Strategy:** Each field and each cube in the simulation is treated as an independent sample in the batch. Total batch size = `number_of_fields * number_of_cubes`.
*   **Batch Limit:** Standalone tests pass at 6000+, but backends (like SmartSim/RedisAI) typically default to a limit of **5000**.

---

## 3. Memory Usage & Size Calculations (fp32)

Calculations are based on 32-bit floats (4 bytes per element).

| Component | Dimensions | Elements | Memory (Bytes) | Memory (Human) |
| :--- | :--- | :--- | :--- | :--- |
| **Input (BS=1)** | `(5, 1, 512)` | 2,560 | 10,240 | ~10 KB |
| **Input (BS=5000)** | `(5, 5000, 512)` | 12,800,000 | 51,200,000 | **~50 MB** |
| **Output (BS=1)** | `(1, 2, 512)` | 1,024 | 4,096 | ~4 KB |
| **Output (BS=5000)**| `(5000, 2, 512)` | 5,120,000 | 20,480,000 | **~20 MB** |

---

## 4. Operational Logic

### **Parallel Independence**
Architecturally, the transformer processes the entire batch dimension **simultaneously**. 
*   There is **no attention/interaction** between different items in the batch.
*   The model effectively makes independent predictions for each cube/field in parallel.
*   The batch dimension must be a multiple of the number of fields (**3**) for the simulation to map outputs back to the grid correctly.

### **Inference Execution**
The model is a forecasting transformer that predicts a sequence of future steps.
*   **Forecast Window:** 2 steps (predicts $t+1$ and $t+2$).
*   **Selection:** The simulation scripts typically use `predictions[-1]`, meaning they discard the immediate next step and use the furthest prediction (the last element of the output sequence) to update the CFD solver.
