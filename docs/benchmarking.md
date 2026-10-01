# Model evaluation

=== "Face Detection"

    We evaluated 13 state-of-the-art face detection algorithms on egocentric video collected from infants and toddlers. Models were evaluated against manually annotated video from both structured head-mounted eye-tracking and naturalistic home-recording contexts.
    Across these datasets, YOLOv11Face (M) and RetinaFace showed the strongest overall agreement with manual annotations. These models are now available on the TinyExplorer App.
    For more details, please check [**Nikolov, T. Y., Yurkovic-Harding, J., Foldes, T., Bradshaw, J., Lai, Y.-K., & D'Souza, H. (2026). Making Machine Learning Accessible for Developmental Science: The Case of Automated Face Detection. Developmental Science, 29(3), e70148.**](https://doi.org/10.1111/desc.70148)

=== "Hand Detection"

    We evaluated 6 open-source hand detection algorithms, with 100 Days of Hands (100DOH) showing the strongest hand-detection performance (presence of hands: 96% precision and 91% recall; hand classification [own or other]: 87% accuracy).
    Beyond model evaluation, we also fine-tuned the 100 Days of Hands on the TinyExplorer data. Both models are available on the TinyExplorer app.

=== "Automatic Speech Recognition"

    Several speech recognition models are available for analysing children's naturalistic speech. We have not evaluated the performance of these models.
    For more information read [Radford et al. (2023), *Robust Speech Recognition via Large-Scale Weak Supervision*](https://arxiv.org/abs/2212.04356)