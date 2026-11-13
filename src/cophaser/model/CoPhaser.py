import torch
import torch.nn as nn
from torch.distributions import Normal
from typing import List
from cophaser._resources import resource_path
from cophaser.model.neuralNet import NeuralNet
from cophaser.model.rhythmic_encoder_VAE import RhythmicEncoderVAE
from cophaser.model.rhythmic_decoder import RhythmicDecoder
from cophaser.model.bernoulli_encoder import RelaxedBernoulliEncoder
from cophaser.loss import Loss
from cophaser import plotting
import anndata
import pandas as pd
from scipy.sparse import csr_matrix
from cophaser import utils
from cophaser.covariates import (
    build_covariate_spec,
    covariate_column_names,
    encode_covariates,
)
import numpy as np
import tqdm


class CoPhaser(nn.Module):
    """
    CoPhaser model combining a rhythmic space and a latent space for non-rhythmic variation, combined using
    a Fourier-based VAE rhythmic decoder, a context-dependant mean shift and amplitude scaling factor.
    """

    def __init__(
        self,
        rhythmic_gene_names: List[str],
        context_genes: List[str],
        n_latent: int = 10,
        n_harm: int = 3,
        rhythmic_decoder_to_all_genes: bool = True,
        use_mu_z_encoder: bool = True,
        use_lambda: bool = True,
        use_latent_z: bool = True,
        rhythmic_encoder_weights=[],
        z_range=20,
        lambda_range=2,
        rhythmic_z_scale=1,
        non_cycling_cells_angle=-1.5,
        force_context_genes_order=None,
        z_input: str = "raw",
    ):
        """
        CoPhaser model combining a rhythmic space and a latent space for non-rhythmic variation, combined using
        a Fourier-based VAE rhythmic decoder, a context-dependant mean shift and amplitude scaling factor.

        Parameters
        ----------
        rhythmic_gene_names: List[str]
            List of rhythmic genes used to learn the rhythmic space f.
        context_genes: List[str]
            List of context genes used to learn the context space z. Usually obtained using the 2000 most variable genes.
        n_latent: int
            Number of latent dimensions for the context space z. Default is 10.
        n_harm: int
            Number of harmonics used to decode the rhythmic contribution. Default is 3.
        rhythmic_decoder_to_all_genes: bool
            Must be True: the rhythmic decoder decodes to all context genes. Kept for saved checkpoints.
        use_mu_z_encoder: bool
            If True, corrects the input of the rhythmic encoder using a mean shift dependent on the latent space z. Default is True.
        use_lambda: bool
            If True, use an amplitude scaling factor dependent on the latent space z. Default is True.
        use_latent_z: bool
            If True, use the latent space z in the generative model. Setting it to False transforms CoPhaser into a PCA-like model. Default is True.
        rhythmic_encoder_weights: List[np.array]
            Pretrained weights for the rhythmic encoder. Can be used to transfer learned weights that performs a 2d projection from rhythmic genes to a f space.
        z_range: float
            Range of the mean shifts derived from z used in the generative model. Default is 20.
        lambda_range: float
            Range of the amplitude scaling factor used in the generative model. Default is 2.
        rhythmic_z_scale: float
            Scaling factor applied to the mean shift derived from z for rhythmic genes. Default is 1.
        non_cycling_cells_angle: float
            Default angle assigned to non-cycling cells in the rhythmic encoder. Default is -1.5, useful when not all cells are cycling (cycling_status_prior of trainer < 1).
        z_input: str
            Input of the context (z) encoder. "raw" (default, original behaviour):
            log1p(counts). "cp10k": log1p(counts / library_size * 1e4), the
            normalisation the rhythmic encoder already uses. Use "cp10k" for deep data
            (bulk, e.g. GTEx with ~1e7-1e8 reads per sample), where raw log counts make
            the z encoder (and its variance) blow up.
        """
        super().__init__()
        if z_input not in ("raw", "cp10k"):
            raise ValueError(f"z_input must be 'raw' or 'cp10k', got {z_input!r}")
        self.z_input = z_input
        if not rhythmic_decoder_to_all_genes:
            raise ValueError("rhythmic_decoder_to_all_genes=False is no longer supported.")
        # store initialization variables
        self.rhythmic_gene_names = rhythmic_gene_names
        if len(self.rhythmic_gene_names) == 0:
            raise ValueError("At least one rhythmic gene must be provided.")
        self.adata_loaded = False
        self.batch_corrected = False
        self.rhythmic_decoder_is_pretrained = False
        self.n_harm = n_harm
        self.n_latent = n_latent
        self.use_mu_z_encoder = use_mu_z_encoder
        self.use_lambda = use_lambda
        self.rhythmic_decoder_to_all_genes = rhythmic_decoder_to_all_genes
        self.z_range = z_range
        self.lambda_range = lambda_range

        # modified by the trainer
        self.cycling_status_prior = 1.0

        # set up variable and rhythmic genes
        if force_context_genes_order is not None:
            self.context_genes = force_context_genes_order
        else:
            self.context_genes = sorted(set(context_genes) | set(rhythmic_gene_names))
        n_variable_genes = len(self.context_genes)
        self.rhythmic_gene_indices = [
            self.context_genes.index(x) for x in self.rhythmic_gene_names
        ]
        self.non_rhythmic_gene_indices = [
            i for i in range(n_variable_genes) if i not in self.rhythmic_gene_indices
        ]

        self.rhythmic_z_scale = torch.ones(len(self.context_genes))
        self.rhythmic_z_scale[self.rhythmic_gene_indices] = rhythmic_z_scale

        # load rhythmic encoder weights if given
        if len(rhythmic_encoder_weights):
            rhythmic_encoder_weights = np.array(rhythmic_encoder_weights)
            assert rhythmic_encoder_weights.shape == (len(self.rhythmic_gene_names), 2)

        # rhythmic autoencoder
        self.rhythmic_encoder = RhythmicEncoderVAE(
            len(self.rhythmic_gene_names),
            self.n_harm,
            preload_weights=rhythmic_encoder_weights,
            default_angle=non_cycling_cells_angle,
        )
        self.rhythmic_decoder = RhythmicDecoder(len(self.context_genes), self.n_harm)

        # variational autoencoder
        self.mean_encoder = NeuralNet(
            n_variable_genes,
            n_latent,
            "none",
        )
        self.var_encoder = NeuralNet(
            n_variable_genes,
            n_latent,
            "exp",
        )

        # bernoulli VAE
        self.cycling_status_encoder = RelaxedBernoulliEncoder(
            len(self.rhythmic_gene_names), n_latent
        )

        # NB/ZINB parameters
        self.log_theta_dispersion = torch.nn.Parameter(torch.randn(n_variable_genes))
        self.dropout_rate = torch.nn.Parameter(torch.tensor(-5.0))

        # delta Mean decoder
        self.decoder_non_rhythmic_contribution = NeuralNet(
            n_latent,
            n_variable_genes,
            "sigmoid",
        )
        self.decoder_raw_output = NeuralNet(n_latent, n_variable_genes, "none")
        self.mu_z_encoder = NeuralNet(
            n_latent,
            len(self.rhythmic_gene_names),
            "sigmoid",
        )

        self.lambda_decoder = NeuralNet(
            n_latent,
            1,
            "sigmoid",
        )

        self._use_latent_z = use_latent_z

        # observed covariates (age, sex, condition, ...): set by add_covariates
        self.covariate_spec = None
        self.covariates = None
        # step 5: covariates also modulate the amplitude of the rhythm (opt-in)
        self.covariate_amplitude = False
        # step 6: covariates also shift the phase of each gene's rhythm (opt-in)
        self.covariate_phase = False
        self.covariate_phase_reference = None
        # if True, the covariate corrections of the rhythmic encoder input (mean shift,
        # log-amplitude) are detached: the covariate effects are then trained by the
        # generative model only, not by the losses on the encoder output (phase prior,
        # entropy, KL of theta, MI when f is not detached). Opt-in, default off.
        self.covariate_encoder_detach = False

    @property
    def n_covariates(self) -> int:
        return 0 if self.covariate_spec is None else len(covariate_column_names(self.covariate_spec))

    @property
    def covariate_names(self) -> List[str]:
        return [] if self.covariate_spec is None else covariate_column_names(self.covariate_spec)

    def add_covariates(
        self, adata, keys, categorical=None, reference=None, amplitude=False, phase=False, phase_reference=None
    ):
        """
        Attach observed covariates from `adata.obs` (one value per cell / sample).

        The encoding rule is learned on the first call and then kept with the model
        (saved in checkpoints): continuous keys are standardised with the training
        mean/sd, categorical keys become one 0/1 column per non-reference level, and
        keys with missing values get an extra `<key>__missing` column. See
        cophaser.covariates. Later calls (or a reloaded model) reuse that rule.

        Parameters
        ----------
        adata: anndata object
        keys: list of adata.obs columns, e.g. ["age", "sex", "condition"].
        categorical: keys to treat as categorical even if numeric (e.g. a 0/1 "sex").
        reference: {key: level} reference level of categorical keys (default: first sorted level).
        amplitude: if True, the covariates also scale the rhythmic part of every gene by
            exp(gamma_g . c) (covariate_amp_decoder, log-amplitude per unit of covariate);
            default False: mesor shift only.
        phase: if True, the covariates also delay the rhythm of every gene,
            F_g(theta - delta_g . c) (covariate_phase_decoder, radians per unit of
            covariate, positive = later peak); default False. Only delays relative to the
            clock are identifiable, see covariate_phase_weight.
        phase_reference: with phase=True, a rhythmic gene whose delay is 0 by definition
            (delays of the other genes are relative to it); default None: delays are
            relative to the amplitude-weighted mean of the rhythmic genes.
        """
        keys = [keys] if isinstance(keys, str) else list(keys)
        if self.covariate_spec is None:
            if phase_reference is not None and phase_reference not in self.rhythmic_gene_names:
                raise ValueError(f"phase_reference {phase_reference!r} is not a rhythmic gene of the model.")
            self.covariate_spec = build_covariate_spec(adata.obs, keys, categorical, reference)
            self.covariate_amplitude = bool(amplitude)
            self.covariate_phase = bool(phase)
            self.covariate_phase_reference = phase_reference if phase else None
            self._build_covariate_layers()
        else:
            known = [s["key"] for s in self.covariate_spec]
            if keys != known:
                raise ValueError(
                    f"The model already encodes covariates {known}; got {keys}. "
                    "The encoding rule is part of the model and cannot change after it is set."
                )
            if bool(amplitude) and not self.covariate_amplitude:
                raise ValueError(
                    "The model was built without covariate amplitude effects; they cannot be "
                    "added afterwards."
                )
            if bool(phase) and not getattr(self, "covariate_phase", False):
                raise ValueError(
                    "The model was built without covariate phase effects; they cannot be "
                    "added afterwards."
                )
        self.covariates = self.encode_covariates(adata)

    def _build_covariate_layers(self):
        """Covariate effect layers, sized from covariate_spec, initialised at 0 so that the
        model starts exactly as without covariates.

        covariate_mean_decoder: W_c, (n_genes x n_covariates), log-mean shift per unit
        of each covariate column, added to the generative mean of every context gene and
        removed from the rhythmic encoder input.
        covariate_amp_decoder (only if covariate_amplitude): W_gamma, (n_genes x
        n_covariates), log-amplitude change of the rhythmic part per unit of each
        covariate column; the rhythmic encoder input is rescaled accordingly.
        covariate_phase_decoder (only if covariate_phase): raw delays (n_genes x
        n_covariates, radians per unit of each covariate column), centred by
        covariate_phase_weight before use. The rhythmic encoder input is not corrected
        (a delay cannot be undone by a per-gene rescaling); use the grid phase
        (grid_phase / Trainer(decoder_phase_from_prior=True)) when delays are large.
        """
        if self.n_covariates == 0:
            return
        device = self.log_theta_dispersion.device
        # skip_init: no random initialisation, so the global torch RNG stream (data
        # shuffling, sampling) is the same as for a model without covariates
        if not hasattr(self, "covariate_mean_decoder"):
            self.covariate_mean_decoder = torch.nn.utils.skip_init(
                nn.Linear, self.n_covariates, len(self.context_genes), bias=False, device=device
            )
            with torch.no_grad():
                self.covariate_mean_decoder.weight.zero_()
        if getattr(self, "covariate_amplitude", False) and not hasattr(self, "covariate_amp_decoder"):
            self.covariate_amp_decoder = torch.nn.utils.skip_init(
                nn.Linear, self.n_covariates, len(self.context_genes), bias=False, device=device
            )
            with torch.no_grad():
                self.covariate_amp_decoder.weight.zero_()
        if getattr(self, "covariate_phase", False) and not hasattr(self, "covariate_phase_decoder"):
            self.covariate_phase_decoder = torch.nn.utils.skip_init(
                nn.Linear, self.n_covariates, len(self.context_genes), bias=False, device=device
            )
            with torch.no_grad():
                self.covariate_phase_decoder.weight.zero_()

    def covariate_mean_shift(self, covariates):
        """W_c . c for all context genes (n_cells x n_genes), or None without covariates."""
        if covariates is None or not hasattr(self, "covariate_mean_decoder"):
            return None
        return self.covariate_mean_decoder(covariates)

    def covariate_log_amplitude(self, covariates):
        """gamma . c for all context genes (n_cells x n_genes), or None without amplitude effects."""
        if covariates is None or not hasattr(self, "covariate_amp_decoder"):
            return None
        return self.covariate_amp_decoder(covariates)

    def covariate_phase_weight(self):
        """Identifiable per-gene delays (n_genes x n_covariates, radians per unit of each
        covariate column), or None without phase effects.

        A delay shared by all genes is the same as a shift of theta, so only differences
        between genes are identifiable. The raw delays are centred on each column: minus
        the delay of covariate_phase_reference if set, otherwise minus the mean over the
        rhythmic genes weighted by their rhythm amplitude in the decoder (norm of their
        Fourier coefficients, not differentiated), so that flat genes, whose phase is
        undefined, do not move the reference.
        """
        if not hasattr(self, "covariate_phase_decoder"):
            return None
        W = self.covariate_phase_decoder.weight
        ref = getattr(self, "covariate_phase_reference", None)
        if ref is not None:
            return W - W[self.context_genes.index(ref)].unsqueeze(0)
        idx = self.rhythmic_gene_indices
        amp = self.rhythmic_decoder.fourier_coefficients.weight.detach()[idx].norm(dim=1)
        w = amp / amp.sum() if float(amp.sum()) > 0 else torch.full_like(amp, 1.0 / len(idx))
        return W - (w @ W[idx]).unsqueeze(0)

    def covariate_phase_shift(self, covariates):
        """delta . c for all context genes (n_cells x n_genes, radians, centred), or None."""
        if covariates is None or not hasattr(self, "covariate_phase_decoder"):
            return None
        return covariates @ self.covariate_phase_weight().T

    def decode_rhythm(self, f, covariates=None):
        """Rhythmic decoder output (n_cells x n_genes) with the covariate delays applied:
        F_g(theta - delta_g . c), i.e. harmonic h of gene g rotated by h . delta_g . c.
        Without phase effects (or covariates) this is exactly rhythmic_decoder(f).

        f: Fourier expansion [cos theta, sin theta, cos 2 theta, ...] (n_cells or 1, 2 n_harm).
        """
        delta = self.covariate_phase_shift(covariates)
        if delta is None:
            return self.rhythmic_decoder(f)
        W = self.rhythmic_decoder.fourier_coefficients.weight
        out = 0
        for h in range(1, self.n_harm + 1):
            c, s = f[:, 2 * h - 2 : 2 * h - 1], f[:, 2 * h - 1 : 2 * h]
            A, B = W[:, 2 * h - 2], W[:, 2 * h - 1]
            cd, sd = torch.cos(h * delta), torch.sin(h * delta)
            # A cos(h(theta - d)) + B sin(h(theta - d))
            out = out + c * (A * cd - B * sd) + s * (A * sd + B * cd)
        return out

    def covariate_effects(self) -> pd.DataFrame:
        """Learned covariate effects as a long table (gene, covariate, kind, value), the
        format of simulation_covariates.CovariateEffects.long(). kind = "mesor": log-mean
        shift per unit of the encoded covariate column (sd units for continuous ones);
        kind = "log_amp" (if covariate_amplitude): log-amplitude change of the rhythm;
        kind = "phase" (if covariate_phase): peak delay in radians (positive = later),
        relative to the clock (see covariate_phase_weight)."""
        if not hasattr(self, "covariate_mean_decoder"):
            return pd.DataFrame(columns=["gene", "covariate", "kind", "value"])
        layers = [("mesor", self.covariate_mean_decoder)]
        if hasattr(self, "covariate_amp_decoder"):
            layers.append(("log_amp", self.covariate_amp_decoder))
        if hasattr(self, "covariate_phase_decoder"):
            layers.append(("phase", None))
        out = []
        for kind, layer in layers:
            weight = self.covariate_phase_weight() if kind == "phase" else layer.weight
            W = weight.detach().cpu().numpy()
            df = pd.DataFrame(W, index=pd.Index(self.context_genes, name="gene"), columns=self.covariate_names)
            long = df.reset_index().melt(id_vars="gene", var_name="covariate")
            long["kind"] = kind
            out.append(long[["gene", "covariate", "kind", "value"]])
        return pd.concat(out, ignore_index=True)

    def encode_covariates(self, adata):
        """Covariate tensor of `adata` with the model's encoding rule (None if the model has none)."""
        if self.covariate_spec is None:
            return None
        return torch.tensor(encode_covariates(adata.obs, self.covariate_spec), dtype=torch.float32)

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.rhythmic_z_scale = self.rhythmic_z_scale.to(*args, **kwargs)

        if getattr(self, "mean_genes", None) is not None:
            self.mean_genes = self.mean_genes.to(*args, **kwargs)

        return self

    def add_batch(self, adata, batch_name):
        batches, _ = pd.factorize(adata.obs[batch_name])
        self.batch_keys = torch.tensor(
            batches,
            dtype=torch.int32,
        )
        self.n_batches = batches.max() + 1
        self.batch_corrected = True

    def add_phase_prior(self, adata, phase_prior_field, phase_prior_kappa=2.0):
        """
        Attach an informative prior on the phase, one angle per cell, from `adata.obs`.

        Like the batch field this is a property of the data rather than of the training run,
        so it is loaded with the counts and travels with the cells through batching.

        ----------
        Parameters
        ----------
        adata: anndata object
            Must carry the prior in `.obs`, in the same cell order as the counts.
        phase_prior_field: str
            `.obs` column holding the prior phase angle in radians. Cells whose value is NaN
            get no prior, so a prior that is only known for part of the sample is allowed.
        phase_prior_kappa: str or float
            How confident the prior is: a `.obs` column for a per-cell concentration, or one
            number for all cells. Larger is tighter (kappa 0 means no prior on that cell, and
            NaN is read as 0). A per-cell kappa is the way to say that some cells are scored
            confidently and others are not.
        """
        if phase_prior_field not in adata.obs:
            raise ValueError(
                f"phase_prior_field '{phase_prior_field}' is not a column of adata.obs."
            )
        angles = np.asarray(adata.obs[phase_prior_field], dtype=float)

        if isinstance(phase_prior_kappa, str):
            if phase_prior_kappa not in adata.obs:
                raise ValueError(
                    f"phase_prior_kappa '{phase_prior_kappa}' is not a column of adata.obs. "
                    "Pass a number instead to use one concentration for every cell."
                )
            kappa = np.asarray(adata.obs[phase_prior_kappa], dtype=float)
        else:
            kappa = np.full(adata.n_obs, float(phase_prior_kappa), dtype=float)
        if np.any(kappa[np.isfinite(kappa)] < 0):
            raise ValueError("phase_prior_kappa must be non-negative.")

        # A cell with no angle cannot have a prior whatever its kappa says, and NaN kappa is
        # read as "no prior" rather than as an error - both are how a partial prior arrives.
        kappa = np.where(np.isfinite(kappa), kappa, 0.0)
        kappa = np.where(np.isfinite(angles), kappa, 0.0)
        angles = np.where(np.isfinite(angles), angles, 0.0)
        # Wrap to (-pi, pi]: the loss only ever uses cos/sin of this, but keeping it canonical
        # makes the stored value comparable with the inferred theta.
        angles = np.arctan2(np.sin(angles), np.cos(angles))

        if not (kappa > 0).any():
            raise ValueError(
                f"No cell has a usable phase prior: every value of '{phase_prior_field}' is "
                "NaN, or the concentration is zero everywhere."
            )
        self.phase_prior = torch.tensor(angles, dtype=torch.float32)
        self.phase_prior_kappa = torch.tensor(kappa, dtype=torch.float32)
        self.phase_prior_loaded = True

    def _get_library_size_rhythmic_var_genes(
        self,
        adata: anndata.AnnData,
        layer_to_use: str,
        library_size_field=None,
    ):

        if not adata.var_names.is_unique:
            raise ValueError(
                "adata.var_names are not unique; call adata.var_names_make_unique() first."
            )
        adata.layers[layer_to_use] = csr_matrix(adata.layers[layer_to_use])
        # Raise an error if some genes are not present in the anndata object
        if not set(self.context_genes).issubset(adata.var_names):
            raise ValueError(
                f"The genes {set(self.context_genes) - set(adata.var_names)} are not present in the anndata object."
            )
        # Extract library size
        if library_size_field is None:
            library_size = adata.layers[layer_to_use].sum(axis=1).A1
            library_size = torch.tensor(library_size, dtype=torch.float32)
        else:
            library_size = torch.tensor(
                adata.obs[library_size_field].to_numpy(dtype=np.float32),
                dtype=torch.float32,
            )
        # transform 0 library sizes to 1 to avoid NaNs
        library_size[library_size == 0] = 1.0
        # Extract rhythmic genes
        rhythmic_genes = (
            adata[:, self.rhythmic_gene_names].layers[layer_to_use].toarray()
        )
        rhythmic_genes = torch.tensor(rhythmic_genes, dtype=torch.float32)
        # variable genes
        variable_genes = adata[:, self.context_genes].layers[layer_to_use].toarray()
        variable_genes = torch.tensor(variable_genes, dtype=torch.float32)
        if (variable_genes < 0).any() or (variable_genes != variable_genes.round()).any():
            raise ValueError(
                f"Layer '{layer_to_use}' contains negative or non-integer values; CoPhaser "
                "expects raw counts (not normalized or log-transformed data)."
            )

        mean_genes = torch.log(
            (variable_genes.mean(axis=0) / library_size.mean())
        ).clamp_min(-50)

        return (library_size, rhythmic_genes, variable_genes, mean_genes)

    def load_anndata(
        self,
        adata: anndata.AnnData,
        layer_to_use: str,
        batch_name: str = None,
        library_size_field: str = None,
        phase_prior_field: str = None,
        phase_prior_kappa=2.0,
        covariate_keys=None,
        categorical_covariates=None,
        covariate_reference=None,
        covariate_amplitude: bool = False,
        covariate_phase: bool = False,
        covariate_phase_reference: str = None,
        covariate_encoder_detach: bool = None,
    ):
        """
        Load anndata object and extract the gene expression matrix, library size, for rhythmic genes and variable genes.

        If a batch_name is given the entropy is computed indep. for each batch, leading to longer training time.

        If the adata contains only a fraction of the genes use the library size field to give the total number of counts in each cell.

        ----------
        Parameters
        ----------
        adata: anndata object
            Anndata object containing the gene expression matrix.
        layer_to_use: str
            The layer to use for gene expression.
        batch: str
            The adata.obs column containing the batch, if any.
        phase_prior_field: str, optional
            The adata.obs column containing a prior phase angle in radians per cell, if any.
            See add_phase_prior.
        phase_prior_kappa: str or float, optional
            Concentration of that prior: an adata.obs column, or one number for every cell.
        covariate_keys: list of str, optional
            adata.obs columns holding observed covariates (age, sex, condition, ...).
            See add_covariates. If the model already has an encoding rule (e.g. it was
            reloaded), it is reused and these keys must match it.
        categorical_covariates: list of str, optional
            Covariate keys to treat as categorical even if numeric.
        covariate_reference: dict, optional
            {key: level} reference level of categorical covariates.
        covariate_amplitude: bool, optional
            The covariates also modulate the amplitude of the rhythm (see add_covariates).
        covariate_phase: bool, optional
            The covariates also delay each gene's rhythm (see add_covariates).
        covariate_phase_reference: str, optional
            Rhythmic gene with delay 0 (default: amplitude-weighted mean of the rhythmic genes).
        covariate_encoder_detach: bool, optional
            If True, the covariate effects removed from the rhythmic encoder input are
            detached there, so that only the generative model trains them (the losses on
            the encoder output no longer leak into the effects). Default None: keep the
            model's current setting (False for a new model, saved in checkpoints).
        """
        if covariate_encoder_detach is not None:
            self.covariate_encoder_detach = bool(covariate_encoder_detach)
        adata = adata.copy()
        (
            self.library_size,
            self.rhythmic_genes,
            self.variable_genes,
            self.mean_genes,
        ) = self._get_library_size_rhythmic_var_genes(
            adata, layer_to_use, library_size_field
        )

        # batch handling
        if batch_name is not None:
            self.add_batch(adata, batch_name)
        else:
            self.batch_corrected = False

        # phase prior handling
        if phase_prior_field is not None:
            self.add_phase_prior(adata, phase_prior_field, phase_prior_kappa)
        else:
            self.phase_prior = None
            self.phase_prior_kappa = None
            self.phase_prior_loaded = False

        # covariates: new keys, or the rule the model already carries
        if covariate_keys is not None:
            self.add_covariates(
                adata, covariate_keys, categorical_covariates, covariate_reference,
                amplitude=covariate_amplitude,
                phase=covariate_phase,
                phase_reference=covariate_phase_reference,
            )
        else:
            self.covariates = self.encode_covariates(adata)

        self.adata_loaded = True

    def prepare_context_training(
        self,
        adata: anndata.AnnData,
        layer_to_use: str,
        batch_name: str = None,
        freeze_mu_z_f_encoder=True,
    ):
        self.rhythmic_encoder.freeze_all_parameters()
        self.rhythmic_decoder.freeze_all_parameters()
        if freeze_mu_z_f_encoder:
            self.mu_z_encoder.freeze_all_parameters()
        self.load_anndata(adata, layer_to_use=layer_to_use, batch_name=batch_name)

    def load_circular_decoder(self, decoder_weights_path):
        self.rhythmic_decoder.load_rhythmic_weights(decoder_weights_path)
        self.rhythmic_decoder.freeze_rhythmic_weights()
        self.rhythmic_decoder_is_pretrained = True

    def latent_encoder_inference(
        self,
        x: torch.Tensor,
        library_size: torch.Tensor,
    ):
        if getattr(self, "z_input", "raw") == "cp10k":
            x_norm = torch.log1p(x / library_size.view(-1, 1) * 1e4)
        else:
            x_norm = torch.log1p(x)
        qz_m = self.mean_encoder(x_norm)
        qz_v = self.var_encoder(x_norm)
        return qz_m, qz_v

    def generative(
        self,
        f: torch.Tensor,
        z: torch.Tensor,
        b_z: torch.Tensor,
        _lambda: torch.Tensor,
        library_size: torch.Tensor,
        f_g1: torch.Tensor,
        covariates: torch.Tensor = None,
    ):
        """
        Generative model of CoPhaser combining the rhythmic contribution decoded from f, the context mean shift decoded from z and the amplitude scaling factor _lambda.
        If b_z is 1, the rhythmic contribution is used, otherwise only the non-rhythmic contribution is used, and the rhythmic contribution is set to the value at f_g1.

        Parameters
        ----------
        f: torch.Tensor
            Rhythmic space representation of the cells.
        z: torch.Tensor
            Context space representation of the cells.
        b_z: torch.Tensor
            Bernoulli variable indicating if the cell is cycling (1) or not (0).
        _lambda: torch.Tensor
            Amplitude scaling factor for the rhythmic contribution.
            Identical to the one used in the rhythmic encoder, and therefore not recomputed here.
        library_size: torch.Tensor
            Library size of the cells.
        f_g1: torch.Tensor
            Values of the rhythmic decoder at the default angle (used for non-cycling cells).
        covariates: torch.Tensor, optional
            Encoded observed covariates (n_cells, n_covariates), see add_covariates.
            They shift the log-mean of every gene by W_c . c (covariate_mean_decoder).
        """
        # context mean shifts
        context_mean_shifts = (
            (
                self.decoder_non_rhythmic_contribution(z) * self.z_range
                - self.z_range / 2
            )
            * int(self._use_latent_z)
            * self.rhythmic_z_scale
        )

        # rhythmic contribution
        # observed covariates: per-gene delay delta . c of the rhythm (step 6)
        F = self.decode_rhythm(f, covariates)
        F_g1 = self.decode_rhythm(f_g1, covariates)
        # observed covariates: per-gene log-amplitude change gamma . c of the rhythm
        covariate_log_amp = self.covariate_log_amplitude(covariates)
        if covariate_log_amp is not None:
            amp = torch.exp(covariate_log_amp)
            F = F * amp
            F_g1 = F_g1 * amp
        F *= _lambda.view(-1, 1)
        F = F * b_z.view(-1, 1) + F_g1.reshape(-1, F.shape[1]) * (1 - b_z).view(-1, 1)

        library_size = library_size.unsqueeze(1)

        delta_mean = F + context_mean_shifts
        # observed covariates: linear log-mean shift W_c . c
        covariate_mean = self.covariate_mean_shift(covariates)
        if covariate_mean is not None:
            delta_mean = delta_mean + covariate_mean
        px_rate = torch.exp(delta_mean + self.mean_genes) * library_size

        # terms required for loss calculation
        log_theta_dispersion_expanded = self.log_theta_dispersion.expand(px_rate.size())
        theta_dispersion = torch.exp(log_theta_dispersion_expanded)

        return {
            "px_rate": px_rate,
            "theta_dispersion": theta_dispersion,
            "F": F,
            "Z": context_mean_shifts,
            "lambda": _lambda,
            "covariate_mean": covariate_mean,
            "covariate_log_amp": covariate_log_amp,
        }

    def forward(
        self,
        variable_genes: torch.Tensor,
        rhythmic_genes: torch.Tensor,
        library_size: torch.Tensor,
        epoch: int = -1,
        use_max_posterior: bool = False,
        covariates: torch.Tensor = None,
        theta_override: torch.Tensor = None,
    ):
        """
        Forward pass of CoPhaser model.

        Parameters
        ----------
        variable_genes: torch.Tensor
            Gene expression matrix of the context genes.
        rhythmic_genes: torch.Tensor
            Gene expression matrix of the rhythmic genes.
        library_size: torch.Tensor
            Library size of the cells.
        epoch: int
            Current epoch of training. Used to adjust the concentration of the rhythmic encoder, and to enable the use of mu_z and _lambda after a certain number of epochs.
        use_max_posterior: bool
            If True, use the maximum of the variational posterior instead of sampling. Used for inference.
        covariates: torch.Tensor, optional
            Encoded observed covariates (n_cells, n_covariates), see add_covariates.
        theta_override: torch.Tensor, optional
            Phase (n_cells,) given to the generative model instead of the encoder's sample,
            e.g. a grid estimate (Trainer(decoder_phase_from_prior=True)). NaN entries keep
            the encoder's phase. The encoder outputs (theta, mu_theta, ...) are returned
            unchanged, so the losses on them still train the encoder.
        """

        # cell specific latent space
        qz_m, qz_v = self.latent_encoder_inference(variable_genes, library_size)
        if not use_max_posterior:
            z = Normal(qz_m, torch.sqrt(qz_v)).rsample() * int(self._use_latent_z)
        else:
            z = qz_m * int(self._use_latent_z)

        # mean shift and amplitude scaling factor from z, for the rhythmic encoder
        mu_z = (self.mu_z_encoder(z) * self.z_range - self.z_range / 2) * int(
            self._use_latent_z
        )

        if self.use_lambda and (epoch > 100 or epoch == -1):
            _lambda = 2 ** (
                self.lambda_decoder(z) * self.lambda_range - self.lambda_range / 2
            )
            lambda_enc = _lambda
        else:
            _lambda = torch.ones_like(qz_m[:, 0])
            lambda_enc = torch.ones_like(qz_m[:, 0])

        # mean shift explained by observed covariates, removed from the rhythmic encoder input
        covariate_shift = self.covariate_mean_shift(covariates)
        if covariate_shift is not None:
            covariate_shift = covariate_shift[:, self.rhythmic_gene_indices]
        covariate_log_amp = self.covariate_log_amplitude(covariates)
        if covariate_log_amp is not None:
            covariate_log_amp = covariate_log_amp[:, self.rhythmic_gene_indices]
        if getattr(self, "covariate_encoder_detach", False):
            # the effects are learned from the generative model only
            if covariate_shift is not None:
                covariate_shift = covariate_shift.detach()
            if covariate_log_amp is not None:
                covariate_log_amp = covariate_log_amp.detach()

        # rhythmic space
        f, theta, mu, kappa, x_projected, f_g1 = self.rhythmic_encoder.forward(
            rhythmic_genes,
            library_size,
            mu_z,
            _lambda,
            self.use_mu_z_encoder and (epoch > 50 or epoch == -1),
            epoch,
            use_max_posterior=use_max_posterior,
            covariate_shift=covariate_shift,
            covariate_log_amp=covariate_log_amp,
        )
        if (epoch > 10) and self.cycling_status_prior < 1:
            b_z, logits = self.cycling_status_encoder.forward(
                rhythmic_genes, z, epoch, library_size
            )
        elif use_max_posterior and self.cycling_status_prior < 1:
            b_z, logits = self.cycling_status_encoder.forward(
                rhythmic_genes, z, 200, library_size
            )  # TODO: modify, the 200 is ugly
        else:
            b_z = torch.ones_like(theta)
            logits = torch.full_like(theta, np.inf)

        if theta_override is not None:
            known = torch.isfinite(theta_override)
            f_override = self.rhythmic_encoder.fourier_basis_expansion(
                torch.where(known, theta_override, torch.zeros_like(theta_override))
            )
            f = torch.where(known.view(-1, 1), f_override, f)

        generative_outputs = self.generative(
            f, z, b_z, _lambda, library_size, f_g1, covariates=covariates
        )
        return generative_outputs, {
            "qz_m": qz_m,
            "qz_v": qz_v,
            "f": f,
            "z": z,
            "b_z": b_z,
            "cycling_logits": logits,
            "theta": theta,
            "mu_theta": mu,
            "mu_z": mu_z,
            "kappa_theta": kappa,
            "lambda_enc": lambda_enc,
            "x_projected": x_projected,
            "f_g1": f_g1,
        }

    def _gene_masks(self, gene_names):
        gene_names = set(gene_names)
        mask_input = [g in gene_names for g in self.rhythmic_gene_names]
        mask_output = [g in gene_names for g in self.context_genes]
        return mask_input, mask_output

    def freeze_genes_rhythmic_VAE(self, gene_names):
        mask_input, mask_output = self._gene_masks(gene_names)
        self.rhythmic_encoder.freeze_weights_genes(mask_input)
        self.rhythmic_decoder.freeze_weights_genes(mask_output)

    def unfreeze_genes_rhythmic_VAE(self, gene_names):
        mask_input, mask_output = self._gene_masks(gene_names)
        self.rhythmic_encoder.unfreeze_weights_genes(mask_input)
        self.rhythmic_decoder.unfreeze_weights_genes(mask_output)

    def _get_gene_annotation(self):
        CCG_path = resource_path("CCG_annotated.csv")
        df_gene = pd.read_csv(CCG_path, index_col="Primary name")["Peaktime"]
        model_gene_names = self.context_genes
        mask = np.isin([g.upper() for g in model_gene_names], df_gene.index)
        model_gene_names = np.array(model_gene_names)[mask]
        categories = df_gene[[g.upper() for g in model_gene_names]]
        return model_gene_names, categories, mask

    def _get_phase_amplitude_mean(self, mask, categories):
        fourier_coeffs = (
            self.rhythmic_decoder.fourier_coefficients.weight.detach()
            .cpu()
            .numpy()[mask][:, [0, 1]]
        )
        phases = np.arctan2(fourier_coeffs[:, 1], fourier_coeffs[:, 0])
        amplitudes = np.sqrt(fourier_coeffs[:, 0] ** 2 + fourier_coeffs[:, 1] ** 2)
        complex_values = amplitudes * np.exp(1j * phases)

        mean_phase = {}
        for cat in np.unique(categories):
            mask = categories == cat
            # mean phase weighted by amplitude
            mean_phase[cat] = np.angle(np.sum(complex_values[mask]))

        return mean_phase, amplitudes, phases

    @staticmethod
    def shift_phases(phases, origin, direction, offset=0):
        return utils.normalize_angles(
            utils.normalize_angles(phases - origin) * direction + offset
        )

    def _get_origine_direction(self, plot=True, offset=1 / 4 * np.pi, ax=None):
        _, categories, mask = self._get_gene_annotation()
        mean_phase, amplitudes, phases = self._get_phase_amplitude_mean(
            mask, categories
        )
        # orientation needs genes annotated as G1/S
        if "G1/S" not in mean_phase:
            return None
        _, _, best_direction = utils.best_order(mean_phase)
        if plot:
            mean_phase_shifted = {
                key: self.shift_phases(
                    value, mean_phase["G1/S"], best_direction, offset=offset
                )
                for key, value in mean_phase.items()
            }
            plotting.plot_phase_distribution(
                mean_phase_shifted,
                amplitudes,
                self.shift_phases(
                    phases, mean_phase["G1/S"], best_direction, offset=offset
                ),
                categories,
                ax=ax,
            )
        return mean_phase["G1/S"], best_direction

    def plot_fourier_coefficients(self, plot_all_genes=False):
        ab_coefficients = (
            self.rhythmic_decoder.fourier_coefficients.weight.detach().cpu().numpy()
        )
        if plot_all_genes:
            gene_names = self.context_genes
        else:
            ab_coefficients = ab_coefficients[self.rhythmic_gene_indices]
            gene_names = self.rhythmic_gene_names
        plotting.plot_fourrier_coefficients(ab_coefficients, gene_names)

    def orient_align_pseudotimes(
        self,
        thetas,
        offset=1 / 4 * np.pi,
        plot=True,
        ax=None,
    ):
        origin_direction = self._get_origine_direction(plot=plot, offset=offset, ax=ax)
        if origin_direction is None:
            return thetas
        origin, direction = origin_direction
        # set the mean g1/s at offset
        thetas = self.shift_phases(thetas, origin, direction, offset=offset)
        return thetas

    def infer_pseudotimes(
        self,
        adata: anndata.AnnData,
        layer_to_use: str,
        isCellCycle=True,
        offset=1 / 4 * np.pi,
        plot=True,
        ax=None,
        library_size_field: str = None,
    ):
        """
        Infer the pseudotimes of the cells in the adata. If isCellCycle, fixes the origin
        of the phase such as G1/S transition arrives at the given offset.
        If the model has covariates, adata.obs must contain them (same encoding rule).
        """
        adata = adata.copy()
        library_size, rhythmic_genes, variable_genes, mean_genes = (
            self._get_library_size_rhythmic_var_genes(
                adata, layer_to_use, library_size_field
            )
        )
        covariates = self.encode_covariates(adata)
        self.to(device="cpu")
        with torch.no_grad():
            _, space_outputs = self.forward(
                variable_genes,
                rhythmic_genes,
                library_size,
                use_max_posterior=True,
                covariates=covariates,
            )
        thetas = space_outputs["theta"]
        if not isCellCycle:
            return thetas
        return self.orient_align_pseudotimes(thetas, offset=offset, plot=plot, ax=ax)

    def _get_posterior_batch_size(self, n_cells, n_genes, device):
        """
        Estimate how many cells get_posterior can process at once without saturating the
        memory available on `device`. Looks at free GPU memory (torch.cuda.mem_get_info)
        for a cuda device, or available system RAM (via psutil, if installed) for cpu.
        Falls back to processing all cells at once if the available memory cannot be
        determined.
        """
        safety_fraction = 0.3
        # rough number of (n_cells x n_genes) float32 tensors alive at once per theta step
        # (px_rate, theta_dispersion, F, rhythmic_expanded, context_mean_shifts, ...)
        tensors_per_step = 10
        bytes_per_cell = max(n_genes, 1) * 4 * tensors_per_step

        available_bytes = None
        if str(device).startswith("cuda") and torch.cuda.is_available():
            free_bytes, _total_bytes = torch.cuda.mem_get_info(device)
            available_bytes = free_bytes
        else:
            try:
                import psutil

                available_bytes = psutil.virtual_memory().available
            except ImportError:
                available_bytes = None

        if available_bytes is None:
            return n_cells

        batch_size = int((available_bytes * safety_fraction) // bytes_per_cell)
        return max(1, min(batch_size, n_cells))

    def get_posterior(
        self,
        adata: anndata.AnnData,
        layer_to_use: str,
        n_points: int,
        use_non_rhythmic=False,
        normalize=True,
        device="cuda" if torch.cuda.is_available() else "cpu",
        library_size_field: str = None,
        batch_size: int = None,
    ):
        """
        Method to compute the exact posterior over a grid of angles for each cell.

        Parameters
        ----------
        adata: anndata.AnnData
            Anndata object containing the gene expression matrix.
        layer_to_use: str
            The layer to use for gene expression.
        n_points: int
            Number of points in the grid to compute the posterior.
        use_non_rhythmic: bool
            If True, includes the non-rhythmic genes in the likelihood computation.
        normalize: bool
            If True, normalizes the posterior to sum to 1.
        device: str
            Device to use for computation.
        library_size_field: str
            As in load_anndata
        batch_size: int
            Number of cells to process at once. If None (default), a batch size is picked
            automatically from the memory currently available on `device` (free GPU memory,
            or system RAM via psutil on cpu) so large datasets don't saturate it.

        Returns
        -------
        posterior: torch.Tensor
            Posterior over the grid of angles for each cell.
        theta_grid: torch.Tensor
            Grid of angles used for the posterior computation.
        """
        adata = adata.copy()
        self.to(device)
        library_size, rhythmic_genes, variable_genes, mean_genes = (
            self._get_library_size_rhythmic_var_genes(
                adata, layer_to_use, library_size_field
            )
        )
        library_size = library_size.to(device)
        variable_genes = variable_genes.to(device)
        rhythmic_genes = rhythmic_genes.to(device)
        covariates = self.encode_covariates(adata)
        if covariates is not None:
            covariates = covariates.to(device)

        n_cells = variable_genes.shape[0]
        if batch_size is None:
            batch_size = self._get_posterior_batch_size(
                n_cells, variable_genes.shape[1], device
            )

        theta_grid = np.linspace(-np.pi, np.pi, n_points)
        res = torch.zeros((n_cells, n_points), device=device)
        with torch.no_grad():
            for start in tqdm.tqdm(range(0, n_cells, batch_size)):
                end = min(start + batch_size, n_cells)
                batch_library_size = library_size[start:end]
                batch_variable_genes = variable_genes[start:end]
                batch_rhythmic_genes = rhythmic_genes[start:end]
                batch_covariates = None if covariates is None else covariates[start:end]

                # use the maximum of the variational posterior of z
                generative, space = self.forward(
                    batch_variable_genes,
                    batch_rhythmic_genes,
                    batch_library_size,
                    -1,
                    use_max_posterior=True,
                    covariates=batch_covariates,
                )
                z = space["z"]
                b_z = space["b_z"]
                _lambda = generative["lambda"]
                f_g1 = space["f_g1"]
                for i, theta in enumerate(theta_grid):
                    thetas = torch.full((z.shape[0],), theta, device=device)
                    f = self.rhythmic_encoder.fourier_basis_expansion(thetas)
                    pred = self.generative(
                        f, z, b_z, _lambda, batch_library_size, f_g1,
                        covariates=batch_covariates,
                    )
                    log_lik = Loss.log_likelihood_NB_weighted(
                        x=batch_variable_genes,
                        px_rate=pred["px_rate"],
                        theta_dispersion=pred["theta_dispersion"],
                        rhythmic_indices=self.rhythmic_gene_indices,
                        non_rhythmic_indices=self.non_rhythmic_gene_indices,
                        rhythmic_likelihood_weight=1,
                        non_rhythmic_likelihood_weight=int(use_non_rhythmic),
                    )
                    res[start:end, i] = log_lik
        theta_grid = self.orient_align_pseudotimes(theta_grid, plot=False)
        if normalize:
            # numerical stability
            res_max, _ = res.max(dim=1, keepdim=True)
            res_stable = res - res_max
            likelihoods = torch.exp(res_stable)
            posterior = likelihoods / likelihoods.sum(dim=1, keepdim=True)
            return posterior.to("cpu"), theta_grid
        return res.to("cpu"), theta_grid

    def grid_phase(
        self,
        n_points: int = 96,
        use_z: bool = False,
        dispersion: float = None,
        kappa_max: float = 50.0,
        batch_size: int = 4096,
        return_loglik: bool = False,
    ):
        """
        Maximum-likelihood phase of every loaded sample on a grid of angles, from the
        rhythmic genes only, with the current rhythmic decoder (no amortised encoder).

        Unlike get_posterior it uses the data already loaded with load_anndata, works
        before training (use_z=False), and returns angles in the decoder's own frame
        (no G1/S re-orientation).

        Parameters
        ----------
        n_points: int
            Number of equally spaced candidate angles in [-pi, pi).
        use_z: bool
            False: log-mean = mean_genes + W_c . c + F(theta) (no context shift, lambda = 1);
            usable before training. True: also add the fitted context shift Z(z) and scale
            F by lambda(z), with z at its posterior mean.
        dispersion: float, optional
            NB dispersion (1 / total_count) for every rhythmic gene. None: the model's
            learned per-gene dispersion (random before training, so pass a value then).
        kappa_max: float
            Upper bound of the returned concentration.
        batch_size: int
            Samples processed at once.
        return_loglik: bool
            Also return the log-likelihood of every sample at every grid angle, e.g. to
            combine samples that share a phase (phase_from_loglik on the summed rows).

        Returns
        -------
        dict of numpy arrays (n_samples,):
            theta: grid angle of maximum likelihood, in (-pi, pi].
            resultant: length of the mean resultant of the normalised grid posterior,
                in [0, 1] (1 = one sharp peak, 0 = flat or two opposite peaks).
            kappa: concentration derived from the resultant (von Mises approximation
                kappa = R (2 - R^2) / (1 - R^2)), clipped to [0, kappa_max]; usable as a
                per-sample confidence for add_phase_prior.
            loglik (only with return_loglik): (n_samples, n_points) log-likelihoods.
            grid (only with return_loglik): (n_points,) grid angles.
        """
        if not self.adata_loaded:
            raise RuntimeError("No data loaded: call load_anndata() first.")
        if dispersion is not None and dispersion <= 0:
            raise ValueError("dispersion must be positive.")
        device = self.log_theta_dispersion.device
        idx = self.rhythmic_gene_indices
        n = self.variable_genes.shape[0]
        grid = torch.linspace(-np.pi, np.pi, n_points + 1, device=device)[:-1]
        if dispersion is None:
            total_count = torch.exp(self.log_theta_dispersion[idx]).view(1, -1)
        else:
            total_count = torch.full((1, len(idx)), 1.0 / dispersion, device=device)
        covariates = getattr(self, "covariates", None)
        loglik = np.empty((n, n_points))
        was_training = self.training
        self.eval()
        with torch.no_grad():
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                x_all = self.variable_genes[start:end].to(device)
                x = x_all[:, idx]
                lib = self.library_size[start:end].to(device).view(-1, 1)
                cov = None if covariates is None else covariates[start:end].to(device)
                base = self.mean_genes.to(device)[idx].view(1, -1).expand(end - start, -1)
                shift = self.covariate_mean_shift(cov)
                if shift is not None:
                    base = base + shift[:, idx]
                lam = torch.ones(end - start, 1, device=device)
                log_amp = self.covariate_log_amplitude(cov)
                if log_amp is not None:
                    lam = lam * torch.exp(log_amp[:, idx])
                if use_z:
                    gen, _ = self.forward(
                        x_all,
                        self.rhythmic_genes[start:end].to(device),
                        self.library_size[start:end].to(device),
                        -1,
                        use_max_posterior=True,
                        covariates=cov,
                    )
                    base = base + gen["Z"][:, idx]
                    lam = lam * gen["lambda"].view(-1, 1)
                tc = total_count.expand_as(x)
                ll = torch.empty(end - start, n_points, device=device)
                for i in range(n_points):
                    f = self.rhythmic_encoder.fourier_basis_expansion(grid[i].repeat(end - start))
                    rate = torch.exp(base + lam * self.decode_rhythm(f, cov)[:, idx]) * lib
                    logits = (rate + 1e-4).log() - (tc + 1e-4).log()
                    ll[:, i] = Loss.log_likelihood_NB(x=x, theta_dispersion=tc, nb_logits=logits)
                loglik[start:end] = ll.cpu().numpy()
        self.train(was_training)
        grid_np = grid.cpu().numpy()
        out = self.phase_from_loglik(loglik, grid_np, kappa_max=kappa_max)
        if return_loglik:
            out.update(loglik=loglik, grid=grid_np)
        return out

    @staticmethod
    def phase_from_loglik(loglik, grid, kappa_max: float = 50.0):
        """
        Phase summary of grid log-likelihoods (rows = samples, or groups of samples whose
        log-likelihoods were summed because they share a phase), as returned by grid_phase:
        theta (argmax), resultant of the normalised posterior, and the von Mises
        concentration kappa = R (2 - R^2) / (1 - R^2) clipped to [0, kappa_max].
        """
        loglik = np.asarray(loglik, dtype=np.float64)
        grid = np.asarray(grid, dtype=np.float64)
        post = np.exp(loglik - loglik.max(axis=1, keepdims=True))
        post /= post.sum(axis=1, keepdims=True)
        resultant = np.clip(np.hypot(post @ np.cos(grid), post @ np.sin(grid)), 0.0, 1.0)
        theta = grid[loglik.argmax(axis=1)]
        r = np.clip(resultant, 0.0, 1.0 - 1e-6)
        kappa = np.clip(r * (2 - r**2) / (1 - r**2), 0.0, kappa_max)
        return {"theta": theta, "resultant": resultant, "kappa": kappa}

    @staticmethod
    def get_variational_posterior(thetas, kappas, n_points=100):
        """
        Compute the variational posterior over a grid of theta values given the parameters of the von Mises distribution.

        Parameters
        ----------
        thetas: torch.Tensor
            Mean directions of the von Mises distributions for each cell. Shape [n_cells].
        kappas: torch.Tensor
            Concentration parameters of the von Mises distributions for each cell. Shape [n_cells].
        n_points: int
            Number of points in the grid to compute the posterior.

        Returns
        -------
        variational_posteriors: torch.Tensor
            Variational posterior over the grid of theta values for each cell. Shape [n_cells, n_points].
        theta_grid: torch.Tensor
            Grid of theta values used for the posterior computation. Shape [n_points].
        """
        thetas = torch.tensor(thetas).clone().detach()
        kappas = torch.tensor(kappas).clone().detach()

        # Grid of theta values: shape [n_points]
        theta_grid = torch.linspace(-np.pi, np.pi, n_points, device=thetas.device)

        # Expand shapes for broadcasting
        mu = thetas.unsqueeze(1)
        theta = theta_grid.unsqueeze(0)

        angle_diff = theta - mu

        # Compute log probs: log P = kappa * log(1 + cos(θ - μ))
        cos_diff = torch.cos(angle_diff)
        log_probs = kappas * torch.log1p(cos_diff)

        # Normalize to get posterior
        log_probs = log_probs - log_probs.max(dim=1, keepdim=True).values  # stability
        probs = torch.exp(log_probs)
        variational_posteriors = probs / probs.sum(
            dim=1, keepdim=True
        )  # [n_cells, n_points]
        return variational_posteriors, theta_grid.flatten()

    def get_outputs(self):
        if not self.adata_loaded:
            raise RuntimeError(
                "Cannot get model output: no data has been loaded. Call load_anndata() first."
            )
        self.eval()
        with torch.no_grad():
            res = self.forward(
                self.variable_genes,
                self.rhythmic_genes,
                self.library_size,
                use_max_posterior=True,
                covariates=getattr(self, "covariates", None),
            )
        return res

    def save(self, file_path: str):
        """Saves the model state and metadata."""
        self.to(device="cpu")
        init_params = {
            "rhythmic_gene_names": self.rhythmic_gene_names,
            "context_genes": self.context_genes,
            "n_latent": self.n_latent,
            "n_harm": self.n_harm,
            "rhythmic_decoder_to_all_genes": self.rhythmic_decoder_to_all_genes,
            "use_mu_z_encoder": self.use_mu_z_encoder,
            "use_lambda": self.use_lambda,
            "use_latent_z": self._use_latent_z,
            "rhythmic_encoder_weights": [],
            "z_range": self.z_range,
            "lambda_range": self.lambda_range,
            "rhythmic_z_scale": self.rhythmic_z_scale[
                self.rhythmic_gene_indices[0]
            ].item(),
            "non_cycling_cells_angle": self.rhythmic_encoder.default_angle,
            "force_context_genes_order": self.context_genes,
            "z_input": getattr(self, "z_input", "raw"),
            "cycling_status_prior": (
                self.cycling_status_prior
                if hasattr(self, "cycling_status_prior")
                else 1
            ),
        }
        torch.save(
            {
                "state_dict": self.state_dict(),
                "init_params": init_params,
                "mean_genes": getattr(self, "mean_genes", None),
                # encoding rule of observed covariates (None if the model has none)
                "covariate_spec": getattr(self, "covariate_spec", None),
                "covariate_amplitude": getattr(self, "covariate_amplitude", False),
                "covariate_phase": getattr(self, "covariate_phase", False),
                "covariate_phase_reference": getattr(self, "covariate_phase_reference", None),
                "covariate_encoder_detach": getattr(self, "covariate_encoder_detach", False),
            },
            file_path,
        )

    @classmethod
    def load(cls, file_path: str):
        """
        Loads the model state and metadata from a file.
        """
        checkpoint = torch.load(file_path, weights_only=False)
        if "init_params" not in checkpoint:
            raise ValueError(
                "Cannot load model: init_params not found in checkpoint. Ensure the model was saved with the save method."
            )
        init_params = dict(checkpoint["init_params"])
        # not a constructor argument: set after construction (absent in old checkpoints)
        cycling_status_prior = float(init_params.pop("cycling_status_prior", 1))
        model = cls(**init_params)
        # absent in checkpoints saved before covariates existed
        model.covariate_spec = checkpoint.get("covariate_spec")
        model.covariate_amplitude = bool(checkpoint.get("covariate_amplitude", False))
        model.covariate_phase = bool(checkpoint.get("covariate_phase", False))
        model.covariate_phase_reference = checkpoint.get("covariate_phase_reference")
        model.covariate_encoder_detach = bool(checkpoint.get("covariate_encoder_detach", False))
        if model.covariate_spec is not None:
            model._build_covariate_layers()  # so that their weights can be loaded
        model.load_state_dict(checkpoint["state_dict"])
        model.cycling_status_prior = cycling_status_prior
        # absent in older checkpoints: then set by load_anndata
        if checkpoint.get("mean_genes") is not None:
            model.mean_genes = checkpoint["mean_genes"]
        return model
