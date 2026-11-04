# A differentiable forward model: parameters and derivatives

Status: branch `differentiable`. The float64 reference `diffractom.differentiable.reference.ReferenceModel`
implements the model below with its analytic derivatives (vector-Jacobian products) for the groups marked
"reference". It reproduces `SinglePhaseForwardOperator` (native projector) to 3e-6, and every derivative agrees with
central finite differences to better than 1e-8 (relative). This was tested with Gauss quadrature + the trapezoid
kernel and with midpoint quadrature + the hat kernel.

## 1. The model

Data $d[a, s, j]$: projection angle $a$ (rotation $\omega_a$), detector bin (translation) $s$, segment $j = cP + p$
(azimuthal bin $c$, ring $p$). With coefficients $c_k(x)$ for the basis orientations $k$:

$$d[a,s,j] = \sum_k \mathrm{PF}[a,k,j]\; S_k[a,s], \qquad S_k[a,s] = \kappa \sum_x K\big(t_a(x) - s\big)\, c_k(x)$$

$$\mathrm{PF}[a,k,j] = I_p\, N(\sigma_k) \sum_{\text{nodes } n} w_n \sum_{h \in \text{ring } p} m_h\; g\big(\mathbf n_{kh}\cdot \mathbf v_{n};\ \sigma_k\big)$$

- kernel $g(d;\sigma) = e^{-(1-|d|)/\sigma^2} + e^{-(1+|d|)/\sigma^2}$ (each term cut where its exponent reaches the cut),
  $N(\sigma) = 1/(8\pi\sigma^2)$;
- pole $\mathbf n_{kh} = U_k \mathbf h$ (distinct symmetry axes $\mathbf h$ of ring $p$ with multiplicities $m_h$);
- probed direction $\mathbf v_n = R(\omega_n)^{\mathsf T}\mathbf u(\eta_n,\theta_p)$,
  $\mathbf u = \cos\theta\,(\cos\eta\,\mathbf d_0 + \sin\eta\,\mathbf d_{90}) - \sin\theta\,\mathbf p_0$, $R(\omega)$ the
  rotation by $\omega$ about $\mathbf k_0$; quadrature nodes $(\omega_n, \eta_n)$ with weights $w_n$ inside the bin;
- projector: $t_a(x) = X_a x + Y_a y + T_{0,a} - \mathrm{cor}/\delta_s + \Delta_a$, trapezoid (pixel footprint) or hat
  kernel $K$, scale $\kappa = \delta_x^2/\delta_s$;
- $I_p$ ring intensity (1 for ring-normalised data, structure factors otherwise).

A loss $L(d)$ (L2, Huber, Poisson deviance with gain and background) enters only through
$r = \partial L/\partial d$. Every parameter gradient below is $\langle r, \partial d/\partial \theta\rangle$, so it works
with any loss.

## 2. What can be differentiated

Notation: $D = \mathbf n\cdot\mathbf v$; $\partial g/\partial D = \operatorname{sign}(D)\, s\,(e_1 - e_2)$ and
$\partial g/\partial s = -(1-|D|)e_1 - (1+|D|)e_2$ with $s = 1/\sigma^2$, $e_{1,2} = e^{-(1\mp|D|)s}$.

| group | parameters (count) | how it enters | derivative | status |
|---|---|---|---|---|
| coefficients | $c_k(x)$ ($K N_x N_y$) | linear | the adjoint $A^{\mathsf T} r$ | operator |
| orientations | rotation $\delta_k$ of every basis orientation ($3K$) | poles: $U_k \leftarrow e^{[\delta_k]_\times}U_k$ | $\partial D/\partial\delta_k = \mathbf n\times\mathbf v$ | reference, GPU (`PFGradient`) |
| kernel widths | $\sigma_k$ ($K$) | $g$ and $N$ | $\partial \mathrm{PF}/\partial\sigma = -\tfrac{2}{\sigma}\mathrm{PF} - \tfrac{2}{\sigma^3} I N\sum w\, m\, \partial g/\partial s$ | reference, GPU |
| Bragg angles | $2\theta_p$ ($P$); from $\lambda$ (1) or lattice parameters (1-6 per phase) | $\mathbf u$ | $\partial\mathbf u/\partial(2\theta) = \tfrac12(-\sin\theta\,\mathbf e - \cos\theta\,\mathbf p_0)$; $\partial(2\theta)/\partial a = -2\tan\theta / a$ | reference, GPU ($2\theta_p$) |
| azimuth zero | $\eta$ offset (1) | $\mathbf u$ | $\partial\mathbf u/\partial\eta = \cos\theta(-\sin\eta\,\mathbf d_0 + \cos\eta\,\mathbf d_{90})$ | reference, GPU |
| rotation axis | tilt of $\mathbf k_0$ about $\mathbf e_1,\mathbf e_2\perp\mathbf k_0$ (2) | $R(\omega) \to Q R Q^{\mathsf T}$ | $\partial\mathbf v/\partial\tau_i = \mathbf e_i\times\mathbf v - R^{\mathsf T}(\mathbf e_i\times\mathbf u)$ | reference, GPU |
| centre of rotation | cor (1) | $t_a$ | $\partial S/\partial\mathrm{cor} = -\tfrac{\kappa}{\delta_s}\sum_x K'(t - s)\,c(x)$ | reference, GPU (`ProjGradient`) |
| per-angle shift | $\Delta_a$ ($N_\Omega$): wobble, run-out, drift | $t_a$ | $\partial S/\partial\Delta_a = \kappa\sum_x K'(t-s)\,c(x)$ | reference, GPU (`ProjGradient`) |
| beam profile | width of the pencil beam (1-2) | Gaussian blur of the prediction along $s$ (commutes with the PF product) | $\partial B/\partial w$ | `BeamBlur` (data_model.py) |
| intensity scales | per ring $r_p$ ($P$), per angle $f_a$ ($N_\Omega$), per translation $t_s$ ($M_y$), per segment efficiency ($N_\eta P$) | $d \to f_a t_s r_p\, d$ | $\langle r, d\rangle$ restricted to the index | `IntensityLayer` (angle, translation, ring; not per segment) |
| background | air $b_j$ + sample part $\beta_j\,\mathcal R_a[\rho](s)$ (2 per segment) | additive | linear | `IntensityLayer` (per segment constant); sample part not built |
| structure factors | Debye-Waller $B$, occupancies (per phase) | $I_p$ via `IntensityModel` | $\partial I_p/\partial B = -2 s_p^2 I_p$ | not built |
| absorption | $\mu(x)$ ($N_x N_y$) | $e^{-\int_{\text{in}}\mu - \int_{\text{out}(\eta,2\theta)}\mu}$ inside the projection | attenuated projector and its adjoint | reference (`AbsorptionReference`, exact to 2e-9), GPU (`AbsorptionOperator`, == reference 3e-6) |
| multi-phase lattice | lattice parameters per phase (1-6), peak widths | window weights $w_{hkl} = \Phi\big(\tfrac{hi - 2\theta_{hkl}}{\varsigma}\big) - \Phi\big(\tfrac{lo - 2\theta_{hkl}}{\varsigma}\big)$ and $\mathbf u$ | chain rule through $2\theta_{hkl}(a, b, c, \alpha, \beta, \gamma)$ | reference (`WindowedReference`, exact to 7e-8), GPU (`WindowedOperator`: family channels + mixing on the GPU, == reference 1e-5), `normalized` True and False; pole directions kept: exact for cubic |
| strain | $\varepsilon(x)$ (6 fields) or per orientation | ring position $2\theta - 2\tan\theta\,\mathbf n^{\mathsf T}\varepsilon\,\mathbf n$ | needs radially resolved data | out of scope here |

### Absorption

`differentiable/absorption.py`: every voxel's contribution is attenuated along the incoming beam and along its
diffracted beam,
$A[a,x,j] = \exp\!\big(-\int_0^\infty \mu(x - t\,\mathbf b_a)\,dt - |k_\parallel|^{-1}\int_0^\infty \mu(x + t\,\mathbf e_{aj})\,dt\big)$.
- $\mathbf e_{aj}$ is the in-plane direction of $k_{out} = \cos 2\theta\,\mathbf p_0 + \sin 2\theta(\cos\eta\,\mathbf d_0 + \sin\eta\,\mathbf d_{90})$,
  and $|k_\parallel|$ its in-plane length (a sample uniform along the axis, as in the NWA 7034 exit-path model).
- The projection becomes $S_k[a,s,j] = \kappa\sum_x K(t-s)\,c_k(x)\,A[a,x,j]$, so it depends on the segment.
- $\partial L/\partial\mu = -\sum_{a,j}(L_a + E_{aj})^{\mathsf T}(A \odot W)$, with
  $W[a,x,j] = \sum_k c_k(x)\,(P_a^{\mathsf T} \mathrm{PF}^{\mathsf T} r)[x,k,j]$.
- Line integrals: on a grid turned to the direction, mu is sampled bilinearly, summed cumulatively along the direction
  and sampled back at the pixels. The transposes are gathers with the same weights.
- GPU (`AbsorptionOperator`): segments are grouped by the in-plane angle of their exit direction (`exit_classes`, bins of
  a few degrees), with attenuation weights W[class, angle, pixel] from the line integrals. Each batch and class gets a
  weighted projection and a PF product masked to the class's segments.
  - The mu gradient contracts the coefficients with the class's $\mathrm{PF}^{\mathsf T} r$ at every angle and pixel,
    then applies the transposed line integrals.
  - It equals the reference with the same classes (forward 3e-6, adjoint 2e-6, mu gradient 2e-6), and with mu = 0 it
    is the plain operator.
  - Cost: n_classes times the projections; the masked PF products read all non-zeros.
- Before applying it to measurements, check on the data which side $\eta = 0$ lies on (left/right intensity ratios).

### Integration windows and lattice parameters

`differentiable/windows.py` (the rock pipelines' windows):
- A window $[lo, hi]$ in $q$ holds the families whose Gaussian profile (width $\varsigma q_f$) reaches into it, with
  overlap $w_f = \tfrac12[\operatorname{erf}(u_{hi}) - \operatorname{erf}(u_{lo})]$,
  $u = (\text{edge} - q_f)/(\sqrt2\,\varsigma q_f)$ and $q_f = 2\pi\sqrt{h^{\mathsf T} G^* h}$.
- The window's pole figure mixes its families' pole figures at the window's $w$-weighted mean $2\theta$, as
  `SinglePhaseForwardOperator(rings=windows_to_rings(...))` does: shares $m_f w_f/\sum m w$ with `normalized=True`,
  and $I_f w_f$ with `normalized=False` ($I_f$ the integrated intensity of the intensity model, held at the built
  lattice: the lattice moves the overlaps and $2\theta$, not $|F|^2$ or the Lorentz factor). The families can be
  given per window (e.g. a pipeline's own selection, `families=`).
- Derivatives with respect to all six lattice parameters (through $\partial G^*/\partial p = -G^*\,\partial G/\partial p\,G^*$)
  and to $\varsigma$.
- The pole directions in the crystal frame are kept fixed. That is exact for cubic phases; for lower symmetry they
  also turn with the lattice, which is not included yet.
- For the rocks: a mineral's composition (solid solutions) moves its reflections between windows, and this makes it
  measurable.
- GPU (`WindowedOperator`): an operator whose data channels are single reflection families, each at its window's
  $2\theta$, plus a mixing layer with the shares. The share derivatives are analytic, and the $2\theta$ part comes from
  `PFGradient` of the family operator. Mixing, its transpose and the per-channel products of the gradient run on the
  GPU (`direct_cl(..., accumulate=True)` for multi-phase sums, a work buffer shared between phases). It costs
  (families / windows) times the PF products of a window operator (NWA 7034: 10-137 families for 48 windows per phase,
  about 5x the window operators in all), so fits use the window operator rebuilt at the current lattice (equal to the
  family form to 3e-6) and the family operator only for the lattice gradient.

### Gauges and identifiability

- A rotation of the whole sample about the rotation axis (an $\omega$ offset) is not identifiable when the
  orientations are free; it is fixed by keeping $\omega$ as given.
- The $\eta$ offset and the axis tilt within the detector plane differ only by a rotation of the sample about the
  beam. With free orientations only their difference is identifiable; with a fixed basis grid both are. In joint fits,
  fix one of them.
- The translation step (voxel size) only rescales the image: not identifiable from these data. The centre of rotation
  is identifiable (the shift that makes $\omega$ and $\omega + 180°$ agree).
- A per-angle intensity scale and per-angle shifts are identifiable from the sample's outline and total intensity.
  They need a smoothness prior if $N_\Omega$ is large compared with the information per angle.
- Bragg angles of ring-integrated data: a shift $\delta(2\theta)$ moves the probe circle on the pole sphere by
  $\delta(2\theta)/2$, which matters once it approaches $\sigma/10$ (0.04° at $\sigma = 0.4°$). Energy or distance
  errors of the integration are of that size; lattice strain ($\delta\theta = \tan\theta\,\varepsilon \approx 10^{-4}$
  rad at $\varepsilon = 10^{-3}$) mostly is not. Strain therefore needs radially resolved data, not ring sums.
- Orientations and coefficients: two basis orientations closer than about $\sigma$ make the split of $c$ between them
  ill-posed, and refining both pulls them together. Merge them, or regularise their distance.

Setting parameters on an operator: `set_pf_parameters` (orientations, widths, and incremental geometry),
`set_geometry` (eta offset, axis tilt, Bragg angles, centre of rotation, per-angle shifts, all relative to the geometry
the operator was built with: for optimisers), `set_proj_parameters` (centre of rotation, per-angle shifts).

## 3. How the derivatives are computed

- **PF parameters** (orientations, $\sigma_k$, $2\theta_p$, $\eta$ offset, axis tilt). With
  $G[a,k,j] = \sum_s r[a,s,j]\,S_k[a,s]$ (the "residual times projection" on the PF support),
  $\partial L/\partial\theta = \sum_{a,k,j} G[a,k,j]\,\partial\mathrm{PF}[a,k,j]/\partial\theta$.
  - On the GPU: one pass over the batches.
    1. Project the batch's coefficients ($S$).
    2. For every stored or generated non-zero of the PF matrix, form $G$ as a dot over the $M_y$ detector bins (the
       sparsity pattern of the forward product).
    3. Evaluate the derivative of the entry from the same loop over nodes and axes as the entry itself.
    4. Sum per orientation (rotation, $\sigma$) or over everything (global parameters).
  - The cost is about one forward pass plus one generation of the PF matrix.
  - The derivative of an entry outside the kernel cut is ignored. At a cut of 4σ these entries are below $e^{-8}$
    of the peak.
- **Projector parameters** (cor, per-angle shifts): with $Z = \mathrm{PF}^{\mathsf T} r$ (the first half of the
  adjoint), $\partial L/\partial\Delta_a = \sum_{s,k} Z[a,s,k]\, \partial S_k[a,s]/\partial\Delta_a$. The derivative
  projection uses $K'$ in place of $K$; $K'$ is piecewise constant for both kernels, so the derivative is exact
  except on the kernel's kinks.
- **Gauss-Newton blocks for orientations.** Ignoring the coupling between orientations,
  $J_k^{\mathsf T} W J_k = \sum_{a,j} \nabla\mathrm{PF}[a,k,j]\,\nabla\mathrm{PF}[a,k,j]^{\mathsf T}\, \sum_s w[a,s,j]\, S_k[a,s]^2$
  ($4\times4$ for rotation and $\sigma$) comes out of the same pass. Each orientation then gets a damped Newton step,
  as in per-grain refinement, but tomographic and jointly with $c$.
- **Few global parameters** (geometry, scales): forward-mode products $\partial d/\partial\theta$ (a sparse product with
  the derivative values) give the Jacobian columns for Gauss-Newton directly.

## 4. What each sample type gains

| sample | most useful parameters | why |
|---|---|---|
| simulated grains (1°, 10° mosaicity) | orientations + $\sigma_k$ | the basis is discrete; refinement moves it to the true orientations and widths (ground truth known: measurable) |
| Al1050 (real) | orientations + $\sigma_k$; axis tilt, $\eta$ offset, cor, $2\theta_p$; ring and angle scales; background | calibration and rotation-stage errors are larger than $\sigma/10$; the misfit (0.4) is far above the noise (0.026) |
| rocks (NWA 7034, meteorite, basalt, ...) | absorption $\mu(x)$; per-phase lattice parameters (composition of solid solutions); geometry; background | transmission down to 6 %, exit-path attenuation 0.5-2x; mineral compositions shift reflections between windows. Measured on NWA 7034: lattice refinement 0.37 % better on held-out angles, absorption 0.07 % (rays T >= 0.3) |
| strain | radially resolved data needed | ring-integrated data are almost insensitive |

## 5. First result: refining the basis orientations and widths (simulated samples)

Setup:
- The indexed basis of the two simulated samples, sigma 0.4°, and an unregularised nonnegative L2 fit.
- Baseline: 200 FISTA iterations from zero.
- Then 10 outer steps. Each is a damped Gauss-Newton step for every orientation (rotation and width: the exact
  4x4 blocks from `PFGradient`, at most 0.2° and 30 % per step, widths kept at or above 0.1°, a line search on the
  loss with the coefficients fixed), followed by 50 FISTA iterations warm-started.
- Control: the baseline continued with FISTA only, 700 iterations in total.
- The quadrature rule was chosen for a width of 0.1°. Script:
  `claude_scratch/diffmodel/refine_sims.py`; V100, 5 s (1°) and 16 s (10°) per outer step.

| sample | median misorientation to the ground truth | p90 | relative residual |
|---|---|---|---|
| 1° mosaicity, K 345 | 0.216° -> **0.097°** (control 0.214°) | 0.42° -> 0.24° | 0.687 -> 0.348 (control 0.685) |
| 10° mosaicity, K 3025 | 0.329° -> **0.282°** (control 0.327°) | 0.64° -> 0.53° | 0.528 -> 0.184 (control 0.520) |

- The orientations moved by 0.24° / 0.37° (median), at most 0.7° / 1.1°.
- 83 % / 57 % of the widths ended at the 0.1° floor: the data want narrower kernels than 0.4°, so the floor is
  binding.
- On the 10° sample the ground-truth score stops improving after three steps while the residual keeps falling. Its
  intragranular orientation gradients are larger than the basis spacing.

### Real data: Al1050 (held-out rotation angles)

- Setup: indexed basis (K 48 017); L2 on 80 % of the angles (every fifth block of 5 held out); the edge-band
  background as a fixed offset; widths kept at or above 0.2°.
- 6 outer steps as above (Gauss-Newton blocks, then 50 FISTA iterations each); L40S.
- Per step: the gradient pass takes 3-4 s, the step with its line search (PF rebuilds) 8-14 s, and FISTA 120 s.

| | train relative residual | held-out relative residual |
|---|---|---|
| baseline, 200 FISTA iterations | 0.3953 | 0.4359 |
| control, 500 FISTA iterations | 0.3755 | 0.4180 |
| refined, 6 steps (500 iterations in total) | **0.3535** | **0.4082** |

- The refined basis predicts the angles the fit did not see 2.3 % better than plain FISTA (5.9 % on the fitted ones).
- The widths settle at a median of 0.28°, the optimum an earlier width sweep on this sample found (0.25-0.3°).

### Geometry recovery (simulated 1° sample)

- Setup: the operator's geometry was perturbed (eta offset +0.2°, axis tilt +0.15° / -0.10°, every Bragg angle
  +0.03°, centre of rotation +0.5 bins) and the coefficients refit.
- Each step: L-BFGS on the 20 geometry parameters (exact gradients, coefficients fixed; about 21 evaluations with PF
  rebuilds, 170 s on a V100), then 50 FISTA iterations.

| after 3 steps | eta offset | tilt | Bragg angles (mean, rms) | centre of rotation |
|---|---|---|---|---|
| remaining error | -0.027° | +0.035°, +0.016° | -0.002°, 0.009° | +0.40 bins (of 0.5) |

- The angular parameters come back to a few hundredths of a degree.
- The centre of rotation barely moves. Each basis orientation diffracts at only a few angles, so its image shifts to
  absorb a constant detector offset. Take the centre of rotation from the transmission or the summed intensity
  instead.

### Geometry on Al1050

- Setup as the Al1050 refinement above (indexed basis, sigma 0.4°, held-out angles); 3 steps of [L-BFGS on the eta
  offset, axis tilt, the 14 Bragg angles and the centre of rotation (15 evaluations, coefficients fixed), 50 FISTA
  iterations]; L40S, about 3 min per lattice-free geometry step.
- The data ask for almost nothing: eta -0.004°, tilt +0.001° / -0.001°, Bragg angles within 0.0005°, centre of
  rotation -0.07 bins. The calibrated geometry is right; there is nothing to gain here.
- (Held out 0.4322 vs 0.4288 for a control of 350 uninterrupted iterations: the refined run restarted FISTA's
  momentum three times; not a geometry effect.)

### Rock sample: NWA 7034 (absorption and lattice parameters)

Setup: scan 0512 at 80 µm (the NWA pipeline's reduction, read only): 8 phases on uniform 15° grids (sigma 8°), 48
windows, the pipeline's weights, rays with transmission >= 0.3, every fifth block of 5 angles held out, FISTA from
zero. Residual: weighted relative L2 (the texture model explains about 15 % of the data's energy at this resolution).

Absorption (50 iterations; `AbsorptionOperator`, attenuation map from the transmission tomogram, 7 exit classes of 4°):

| model | train | held out |
|---|---|---|
| no correction | 0.9297 | 0.9253 |
| the pipeline's per-ray exit-path factors | 0.9288 | 0.9246 |
| voxel-resolved absorption | 0.9288 | 0.9247 |
| voxel-resolved, exit side mirrored | 0.9314 | 0.9272 |

- The voxel model and the pipeline's per-ray approximation agree (1e-4); both are 0.07 % better than no correction,
  and the mirrored exit side is clearly worse: the data confirm the exit-side convention.
- With rays below 30 % transmission left out the correction hardly changes the fit; its value is in the darker rays
  and in phase amounts (not scored here). The voxel model costs about 4x per iteration.

Lattice parameters (`WindowedOperator`, normalized=False, the pipeline's families): 26 parameters with the
crystal-system constraints (triclinic plagioclase 6 ... cubic magnetite 1, lengths +-2 %, angles +-1°) and one
relative peak width; 3 steps of [L-BFGS-B, 15 evaluations at fixed coefficients (20 s each on a V100), then 40 FISTA
iterations with the window operators rebuilt at the new lattice]; control: the same FISTA schedule (restarts
included) at the CIF lattices.

| iterations | refined: train / held out | control: train / held out |
|---|---|---|
| 50 (start) | 0.9297 / 0.9253 | 0.9297 / 0.9253 |
| 90 | 0.9251 / 0.9219 | 0.9274 / 0.9238 |
| 130 | 0.9220 / 0.9198 | 0.9256 / 0.9227 |
| 170 | 0.9197 / **0.9185** | 0.9242 / **0.9219** |

- The refined lattices predict the held-out angles better, by 0.0034 after three steps (0.7 % of the residual
  energy, 4 % more explained energy), and the gap grows with every step.
- The peak width stays at the pipeline's value (0.00500 -> 0.00501).
- Some changes are stable from step to step: oligoclase c +0.6-0.7 %, orthopyroxene b -0.63..-0.67 %, magnetite a
  +0.22-0.28 %, ilmenite a +0.6-0.8 % and c -0.64..-0.68 %. Others drift to the bounds: pigeonite b +2 %, augite c
  -2 %, zircon a -1.45 %. Those are not plausible compositions. The three pyroxenes overlap heavily, and a window of
  +-1.25 FWHM is sensitive to a peak position mainly near its edges, where moving a reflection redistributes
  intensity between neighbouring windows. So the lattice parameters also act as intensity knobs.
- To measure lattices rather than fit them: finer q bins (or the radial profile) in the windows' place, a prior
  towards the CIF values with a plausible spread, or fewer parameters (one isotropic scale per phase).
- Scripts: `claude_scratch/diffmodel/nwa` (absorption_test.py, lattice_stage.py, summary.md).

## 6. Suggested order

1. GPU vector-Jacobian product for the PF parameters (orientations, $\sigma_k$, global geometry), checked against the
   reference. Then alternate coefficient and orientation/width steps (block Gauss-Newton) on the simulated samples,
   scored against the ground truth.
2. Projector parameters (cor, per-angle shifts) on the GPU (derivative projector), plus intensity scales and
   background; geometry refinement on Al1050.
3. Window weights from a peak-profile model with lattice-parameter derivatives (multi-phase rocks).
4. Absorption as a model term (attenuated projector), for the rocks.
