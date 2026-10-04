# References

Consolidated from `LITERATURE_REVIEW.md` and `DATASET_SELECTION.md`. Full annotated
entries (methodology/dataset/results/limitations/relevance) live in those files; this
is the citation list in a single place, as the project structure requires.

1. Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory. *Neural Computation*, 9(8), 1735–1780.
2. Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. *Proceedings of NAACL-HLT 2019*, pp. 4171–4186. arXiv:1810.04805.
3. Ji, S., Zhang, T., Ansari, L., Fu, J., Tiwari, P., & Cambria, E. (2022). MentalBERT: Publicly Available Pretrained Language Models for Mental Healthcare. *Proceedings of LREC 2022*.
4. Cao, Y., Dai, J., Wang, Z., Zhang, Y., Shen, X., Liu, Y., & Tian, Y. (2024). Machine Learning Approaches for Mental Illness Detection on Social Media: A Systematic Review of Biases and Methodological Challenges. arXiv:2410.16204.
5. Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., et al. (2021). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. *ICLR 2021*. arXiv:2010.11929.
6. Agung, E. S., Rifai, A. P., & Wijayanto, T. (2024). Image-based facial emotion recognition using convolutional neural network on emognition dataset. *Scientific Reports*, 14. https://doi.org/10.1038/s41598-024-65276-x
7. Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra, D. (2017). Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization. *ICCV 2017*.
8. Niu, T., Zhu, S., Pang, L., & El Saddik, A. (2016). Sentiment Analysis on Multi-View Social Data. *Proceedings of the International Conference on Multimedia Modeling (MMM 2016)*. [MVSA dataset origin — primary dataset of this project]
9. Gandhi, A., Adhvaryu, K., Poria, S., Cambria, E., & Hussain, A. (2023). Multimodal sentiment analysis: A systematic review of history, datasets, multimodal fusion methods, applications, challenges and future directions. *Information Fusion*, 91, 424–444.
10. Das, R., & Singh, T. D. (2023). Multimodal Sentiment Analysis: A Survey of Methods, Trends, and Challenges. *ACM Computing Surveys*, 55(13s). https://doi.org/10.1145/3586075
11. Jiang, M., & Ji, S. (2022). Cross-Modality Gated Attention Fusion for Multimodal Sentiment Analysis. arXiv:2208.11893.
12. Alam, F., Ofli, F., & Imran, M. (2018). CrisisMMD: Multimodal Twitter Datasets from Natural Disasters. *Proceedings of ICWSM 2018*. arXiv:1805.00713.
13. Vasanthi, P., & Viswanatham, V. M. (2026). Multimodal sentiment analysis: hybrid classification model with image and text feature descriptors. *Scientific Reports*, 16, Article 13987. https://doi.org/10.1038/s41598-026-42912-2
14. Li, Z., Xu, B., Zhu, C., & Zhao, T. (2022). CLMLF: A Contrastive Learning and Multi-Layer Fusion Method for Multimodal Sentiment Detection. *Findings of the Association for Computational Linguistics: NAACL 2022*, pp. 2282–2294. https://aclanthology.org/2022.findings-naacl.175 — [source of this project's actual MVSA-Single train/dev/test split and label mapping; see `DATASET_SELECTION.md`]

## Dataset acquisition sources (not academic citations, but load-bearing for reproducibility)
- HuggingFace Hub dataset `xwycyj/MVSA-Single` — https://huggingface.co/datasets/xwycyj/MVSA-Single (community re-upload used after the official MVSA OneDrive link returned 404; content-verified, see `DATASET_SELECTION.md`).
- GitHub repository `Link-Li/CLMLF` — https://github.com/Link-Li/CLMLF (official code release for reference 14 above; source of the labels/split actually used).

## Sharma et al. (2020) — SemEval-2020 Task 8 (Memotion), documented fallback, not used
Sharma, C., Bhageria, D., Scott, W., Pykl, S., Das, A., Chakraborty, T., Pulabaigari, V., & Gambäck, B. (2020). SemEval-2020 Task 8: Memotion Analysis - the Visuo-Lingual Metaphor! *Proceedings of SemEval-2020*, ACL Anthology 2020.semeval-1.99. Documented in `DATASET_SELECTION.md` as the fallback dataset; **not actually used**, since the primary MVSA acquisition succeeded.
