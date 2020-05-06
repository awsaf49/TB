# A Composite Neural Network for Tuberculosis Diagnosis from Chest X-rays with Deep Feature Fusion Scheme using Multi-stage Transfer Learning with Layerwise Fine Tuning

Tuberculosis, a life-threatening disease, has become
one of the major threats for public health mostly in developing
countries due to the lack of experts. Significant research has
been carried out in automating the diagnosis of tuberculosis
due to the difficulty of traditional diagnostic measures. In this
paper, a deep neural network-based approach is proposed for
tuberculosis diagnosis using chest X-ray images. Firstly, a multistage transfer learning approach is introduced to effectively
utilize the learning on other larger datasets ordered according
to relevance for tuberculosis diagnosis. Moreover, a layerwise
fine-tuning approach is proposed to effectively fine-tune a very
deep network using a smaller number of data. Additionally,
a very efficient and lightweight custom network architecture
is also proposed that incorporates features from a broader
spectrum. Finally, a deep feature fusion scheme is proposed
to combine the extracted features of different networks for
further joint optimization of the extracted feature spaces. A
gradient-based discriminative localization is also integrated with
the proposed approach to visually analyze the significant portions
of the images that instigated the decision. This approach will
provide the opportunity to effectively utilize learning of other
relevant applications to explore the available training data for
precise diagnosis of tuberculosis. Extensive experimentations on
three publicly available databases containing chest X-rays of
tuberculosis patients provide significant performance in all traditional evaluation metrics outperforming other state-of-the-art
approaches.

<img align="center" src="https://github.com/awsaf49/TB/blob/master/Figures/grad-cam.png" width="900">
<img align="center" src="https://github.com/awsaf49/TB/blob/master/Figures/multitf.PNG" width="900">
<img align="center" src="https://github.com/awsaf49/TB/blob/master/Figures/network.png" width="900">
<img align="center" src="https://github.com/awsaf49/TB/blob/master/Figures/layerwise.PNG" width="900">
<img align="center" src="https://github.com/awsaf49/TB/blob/master/Figures/ensemble.PNG" width="900">
