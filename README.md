# TripleAgonist_MCNN: A Computational Workflow for Prioritizing Candidate Peptides Targeting GLP-1R, GIPR, and GCGR

Yu-Han Weng, Yi-Ting Chung, Sin-Siang Wei, Wei-En Jhang, Yi-Chen Hsu, Yu-Yen Ou

| [🎇 Abstract](#abstract) | [📃 Dataset](#Dataset) | [💾 Requirements](#requirement) | [📚 License](#License) |
| --- | --- | --- | --- |

## 🎇Abstract <a name="abstract"></a>

TripleAgonist_MCNN is a structure-generative and sequence-discriminative computational workflow for prioritizing candidate peptides that simultaneously activate the glucagon-like peptide-1 receptor (GLP-1R), glucose-dependent insulinotropic polypeptide receptor (GIPR), and glucagon receptor (GCGR). The pipeline generates 39-residue peptide backbones via RFdiffusion conditioned on receptor-bound PDB templates (`7FIY`, `7FIM`, and `7V35`) and designs 864,000 C-terminal sequences using ProteinMPNN across a temperature gradient ($T = 0.1\text{–}1.0$). Candidate sequences are encoded with frozen ProtTrans embeddings (with the conserved N-terminal 15-residue activation domain excised at the embedding level) and filtered through three independent receptor-specific multi-window convolutional neural networks (MCNN; kernel widths 2, 4, 6, 8, 10, 12, 14, and 16). Across 5-fold cross-validation, the MCNN classifiers achieved mean AUCs of 0.9483 (GLP-1R), 0.9861 (GCGR), and 0.9757 (GIPR).

TripleAgonist_MCNN yields a refined set of N-terminally standardized triple-agonist candidates validated via AlphaFold3 complex structure prediction ($\text{pTM} > 0.5$, $\text{ipTM} > 0.7$, interfacial $\text{PAE} < 5\text{ \AA}$) to support downstream experimental synthesis and screening. Predictions reflect computational sequence classification and structural plausibility rather than direct experimental confirmation of receptor agonism or therapeutic efficacy. <br>

## 📃Dataset <a name="Dataset"></a>

Primary peptide sequences for GLP-1, GCG, and GIP were retrieved from the ChEMBL database and deduplicated to construct receptor-specific binary classification datasets. Positive samples comprise peptides experimentally validated to activate the target receptor, whereas negative samples consist of peptides from the other two families known not to bind or activate that receptor.

| Dataset | Split | Positive Sequences | Negative Sequences |
| :--- | :---: | :---: | :---: |
| GLP-1 | Train | 350 | 53 |
| GLP-1 | Test | 87 | 13 |
| GCG | Train | 126 | 252 |
| GCG | Test | 31 | 63 |
| GIP | Train | 37 | 338 |
| GIP | Test | 9 | 85 |

| Virtual Screening Stage | GLP-1R Track (`7FIM`) | GIPR Track (`7FIY`) | GCGR Track (`7V35`) | Total |
| :--- | :---: | :---: | :---: | :---: |
| Backbones Generated (RFdiffusion) | 10 | 10 | 10 | 30 |
| Sequences Designed (ProteinMPNN) | 288,000 | 288,000 | 288,000 | 864,000 |
| Triple-MCNN Intersection Hits\* | 2 | 3 | 6 | 11 |
| AlphaFold3 Structural Validation Passed | 1 | 2 | 2 | 5 |

*\*Prior to AlphaFold3 structural validation, the N-terminal 15 residues of the 11 triple-MCNN intersection hits were standardized to the native GIP N-terminal sequence.*

## 💾Requirements <a name="requirement"></a>

```bash
h5py==3.15.0
tqdm==4.67.1
numpy==2.3.3
scikit-learn==1.7.2
tensorflow==2.20.0