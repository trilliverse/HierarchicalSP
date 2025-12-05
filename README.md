# S&P: Towards Scalable and Powerful Graph Learning with Hierarchical Structural Acquisition

<p align="center">
  <a href="https://www.gnu.org/licenses/gpl-3.0.html">
    <img src="https://img.shields.io/badge/License-GPLv3-blue.svg" alt="GPLv3 License">
  </a>
</p>
This is the official implementation of the paper "S&P: Towards Scalable and Powerful Graph Learning with Hierarchical Structural Acquisition".

---

## Quickstart

1. **Configure hyper-parameters**

   Edit `config.yaml` to set dataset names, model structure, and training options.
    
2. **Preprocess data**

   Generate structural features and serialized `.pt` artifacts:

   ```bash
   python preprocess.py --dataset <dataset_name>
   ```

   Outputs are saved to `processed/<dataset>_*.pt` based on the configuration flags.

3. **Train and evaluate**

   ```bash
   python main.py --dataset <dataset_name>
   ```