# 【MSSP 2025】DWCN-SDGFD code

---

**This is the source code for "Discrete Wavelet Convolutional Network with Cross-Contrast Perturbation for Single Domain Generalization in Fault Diagnosis". You can refer to the following steps to reproduce the single domain generalization of the fault diagnosis task.**

# :triangular_flag_on_post:Highlights

----

**A new single-domain generalization fault diagnosis method is introduced, leveraging a learnable wavelet network to enhance noise resistance and enabling robust feature perturbation while maintaining semantic consistency, thus improving model generalization.**

**A trainable discrete wavelet convolutional network is designed to suppress noise and extract robust multi-frequency features through optimized wavelet kernels.** 

**A cross-contrast perturbation strategy enhances feature diversity by maximizing domain differences while preserving semantic consistency.**

**An instance similarity learning mechanism refines feature space consistency and sharpens class boundaries to improve generalization.**

# ⚙️Abstract

----

**Single-source domain generalization for fault diagnosis aims to train on a single-condition source domain and generalize to multiple unknown target domains, drawing widespread attention. Existing methods typically generate new domain features to expand data distribution and extract domain-invariant representations, but still face key challenges: (1) It is difficult for the model to extract robust features under noisy conditions, and feature perturbation may amplify the noise components, resulting in unreliable representation results; (2) Domain expansion often relies on random perturbations without proper constraints, leading to semantic distortion of the generated features; and (3) domain invariant feature learning overemphasizes semantic alignment while overlooking instance-level learning, failing to capture fine-grained inter-instance differences. To cope with these challenges, we propose a cross-contrast perturbation single-domain generalization architecture based on discrete wavelet convolutional networks. First, learnable discrete wavelet operators are employed to extract noise-robust features. Then, cross-contrast perturbation between the main and auxiliary networks enhances feature diversity while preserving semantic consistency, thereby improving the reliability of domain extension. Instance similarity learning is further introduced to ensure consistent feature space characterization and enhance model generalization. Finally, extensive experiments were conducted on gearbox and bearing datasets to validate the effectiveness of the proposed method.**

# :blue_book:Proposed Method

---

![2.png](https://youke1.picui.cn/s1/2025/09/19/68cd5dc7c61d3.png)

# 📄Citation

## If you find this paper and repository useful, please cite us! ⭐⭐⭐

----

```
@article{wei2025discrete,
  title={Discrete wavelet convolutional network with cross-contrast perturbation for single domain generalization in fault diagnosis},
  author={Wei, Jiacheng and Wang, Qibin and Zhang, Guowei and Wang, Yi and Zhao, Haonan},
  journal={Mechanical Systems and Signal Processing},
  volume={239},
  pages={113286},
  year={2025},
  publisher={Elsevier}
}
```



