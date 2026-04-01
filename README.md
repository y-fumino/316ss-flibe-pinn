# 316SS-FLiBe-PINN

Multi-Physics Coupled Physics-Informed Neural Networks for Inverse Estimation of Chromium Diffusion Parameters in 316 Stainless Steel Corroded by Molten FLiBe Salt.

## Overview

This repository contains the source code and extracted experimental data for the paper:

**"Quantifying Grain Boundary and Bulk Diffusion Contributions in 316 Stainless Steel / FLiBe Corrosion via Multi-Physics Physics-Informed Neural Networks"**

The coupled PINN framework simultaneously assimilates spatial (EDS line scan) and temporal (gravimetric mass loss) experimental data through an auxiliary-variable-based integral mass conservation constraint to estimate:

- **D₀,PINN** = 4.3 × 10⁻¹⁴ cm²/s (base effective diffusion coefficient)
- **α** = −0.56 (concentration-dependence parameter)

## Repository Structure

```
316ss-flibe-pinn/
├── README.md
├── requirements.txt
├── data/
│   ├── spatial_Cr_profile_700C_3000h.csv     # Cr concentration vs depth (48 points)
│   └── mass_loss_700C.csv                     # Weight loss at 1000, 2000, 3000 h (6 points)
├── src/
│   ├── coupled_pinn_316ss.py                  # Main coupled PINN model
│   └── plot_figures.py                        # Figure generation for the paper
└── LICENSE
```

## Data Sources

All experimental data were extracted from published literature using WebPlotDigitizer:

- **Spatial concentration profile**: J. Nucl. Mater. 482 (2016) 147–155, Figure 3(a). 316SS in 316SS crucible, purified FLiBe, 700 °C, 3000 h.
- **Mass loss data**: Ph.D. Thesis, University of Wisconsin–Madison (2015), Figure 75. 316SS in 316SS crucible, purified FLiBe, 700 °C, 1000–3000 h.

## Requirements

- Python 3.8+
- DeepXDE 1.x
- PyTorch 1.x or 2.x
- NumPy
- Matplotlib

Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Run the coupled PINN model
python src/coupled_pinn_316ss.py

# Generate paper figures (after model training)
python src/plot_figures.py
```

## Governing Equation

One-dimensional nonlinear diffusion with concentration-dependent diffusion coefficient:

∂C/∂t = ∂/∂x [D(C) ∂C/∂x]

where D(C) = D₀(1 + α(1 − C_norm))

The mass conservation constraint is incorporated via an auxiliary variable w:

∂w/∂x = 1 − u(x, t)

connecting the spatial concentration profile to gravimetric mass loss.

## Citation

If you use this code, please cite:

```
Y. Fumino, Quantifying Grain Boundary and Bulk Diffusion Contributions in 316 Stainless 
Steel / FLiBe Corrosion via Multi-Physics Physics-Informed Neural Networks, 
Journal of Nuclear Materials (submitted).
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
