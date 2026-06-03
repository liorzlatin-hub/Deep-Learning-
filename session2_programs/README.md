# Deep Learning Session 2 - Python Programs

This folder translates the topics listed in **Deep Learning Session 2.pdf** into standalone Python scripts.

## Why this folder exists
The PDF text extraction is sparse (most slides are visual), but it clearly references these notebook themes:
- Advanced tensors (shape/stride/contiguity)
- Tensor serialization
- Tensor images
- Tensor time series
- Complete enumeration
- Temperature/linear gradient descent
- Manual gradient descent + normalization
- Quadratic gradient descent
- PyTorch autograd

Instead of notebooks, you asked for Python programs, so each script is a clean `.py` equivalent.

## Program map and rationale
1. `01_tensor_advanced.py`
- Why created: Implements advanced tensor basics from the slide topic "Tensors - Advanced".
- What it teaches: shape, strides, transpose behavior, and contiguity.

2. `02_tensor_serialization.py`
- Why created: Implements "Tensors - Serialization" in script form.
- What it teaches: saving/loading arrays and metadata reproducibly.

3. `03_tensor_images.py`
- Why created: Implements "Tensor Images" topic from the session list.
- What it teaches: image tensor channel handling, normalization, and grayscale conversion.

4. `04_tensor_time_series.py`
- Why created: Implements the "Tensor - Time Series" notebook idea.
- What it teaches: turning a sequence into supervised windows for ML.

5. `05_complete_enumeration_vs_gd.py`
- Why created: Implements the "complete_enumeration" concept.
- What it teaches: brute-force search compared with gradient descent.

6. `06_manual_gradient_descent_normalization.py`
- Why created: Implements "manual_gradient_descent_normalization".
- What it teaches: why normalization helps optimization convergence.

7. `07_quadratic_gradient_descent.py`
- Why created: Implements "quadratic_gradient_descent".
- What it teaches: sensitivity to learning rate and start point on a quadratic objective.

8. `08_autograd_gradient_descent.py`
- Why created: Implements "pytorch_autograd" from session topics.
- What it teaches: automatic differentiation for parameter updates.

## Plotting versions (added)
To support visual learning, plotting versions were added for optimization-focused scripts:

1. `05_complete_enumeration_vs_gd_plot.py`
- Why created: Visualize convergence and fitted line for the enumeration-vs-GD topic.
- Output: `artifacts/05_convergence.png`, `artifacts/05_fit.png`.

2. `06_manual_gradient_descent_normalization_plot.py`
- Why created: Make the normalization effect visible by plotting both loss curves together.
- Output: `artifacts/06_normalization_convergence.png`.

3. `07_quadratic_gradient_descent_plot.py`
- Why created: Show how initialization and learning rate change optimization trajectory on a quadratic objective.
- Output: `artifacts/07_quadratic_trajectories.png`.

4. `08_autograd_gradient_descent_plot.py`
- Why created: Plot autograd training loss over epochs.
- Output: `artifacts/08_autograd_convergence.png`.

## Notebook conversions (added)
- Matching notebooks were generated in `notebooks/` (one notebook per numbered script).
- Purpose: same content as scripts, but in a notebook format for step-by-step experimentation.

## Torch installation and validation (added)
- `torch` was installed in the local virtual environment.
- `08_autograd_gradient_descent.py` was run and validated on the torch backend.
- `08_autograd_gradient_descent_plot.py` was also run and generated the expected plot artifact.

## Setup
From this folder:

```powershell
pip install -r requirements.txt
```

Then run any script, for example:

```powershell
python 05_complete_enumeration_vs_gd.py
```

Or run everything in one command:

```powershell
python run_all.py
```

From the workspace root, you can run it with:

```powershell
c:/Users/liorz/VSCode_Projects/deep_learning/.venv/Scripts/python.exe .\session2_programs\run_all.py
```

## Notes
- These scripts are intentionally small and educational.
- They are organized so you can study one concept per file.
