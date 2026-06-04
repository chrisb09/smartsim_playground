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

## 3. Summary Table
| Input | Description | Typical Value/Shape |
| :--- | :--- | :--- |
| **`src`** | Past sequence of flow states | `(5, Batch, 512)` |
| **`seq_len`** | Number of past time steps | 5 |
| **`feature_dim`** | Size of flattened 3D cube | 512 ($8^3$) |
| **`batch_size`** | Fields $\times$ Cubes | Variable (e.g., $3 \times 100$) |
| **`forecast_window`**| Number of future steps to predict | 2 |
