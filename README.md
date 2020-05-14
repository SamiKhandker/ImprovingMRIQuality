The directories here are split as such:
* deepinpy_documentation: contains all newly generated documentation for the DeepInPy research project
* deepinpy_unet_implementation: contains all code related to the unet implementation added to deepinpy
* mcmri_preprocessing: Contains all code related to preprocessing the input data
* normalizing_flows_model: Contains all code related to the Normalizing Flows model


https://medium.com/@jaxon.lightfoot/improving-undersampled-mri-with-deep-learning-bf991672b006




# Improving Undersampled MRI with Deep Learning

Comparing Normalizing Flows and Deep Basis Pursuit

Angelos Georgios Koulouras, Jackson Lightfoot, Sami Khandker, Tarek Allam, Menelaos Kaskiris

## Introduction and Background

Magnetic resonance imaging (MRI) is one of the most important medical imaging techniques in today’s time. MRI scans help medical professionals diagnose a wide variety of conditions and are particularly useful for examining the human nervous system. Unfortunately, MRI scans are both costly and time consuming, so reducing the number of samples taken during a scan is a very relevant problem.

When a scan is performed, a magnetic field is used to align hydrogen atoms in the human body. A radio frequency pulse is then emitted through a cross section of the body, and the signal is recorded by receiver coils. The data recorded by these coils is in the frequency domain, known as k-space. Taking the inverse Fourier transform of k-space results in an image, representative of the concentration of hydrogen atoms in that cross section of the body.

The forward sensing model for this process involves the image and sensitivity profiles for each coil. In the toy example of 3 coils shown below, pointwise multiplication of the image times each sensitivity map results in 3 new images. Applying the discrete Fourier transform (DFT) on each sensitivity-calibrated image results in 3 k-space grids.

![](https://cdn-images-1.medium.com/max/2000/0*V73GnLml9eu4V4_F)

The resolution of an MRI image is highly dependent on the number of k-space measurements. Compressed sensing aims to reduce the number of k-space measurements while still maintaining a high quality image. This is done via incoherently subsampling the k-space, which introduces noise-like aliasing when performing the adjoint (backwards) transform.

Compressed sensing MRI is a classic example of an inverse problem. An inverse problem occurs when the output of a physical system is measured to model the input or state of the system. In the case of compressed sensing MRI, the state of the system being modeled is the concentration of hydrogen atoms in a cross section of the body, while the measured output is the subsampled k-space.

![](https://cdn-images-1.medium.com/max/2000/0*KE-ImRIP3liEN34Q)

![](https://cdn-images-1.medium.com/max/2000/0*-EQrStzpYsnshP8J)

In the adjoint sensing model for compressed sensing, each k-space channel is pointwise multiplied by a mask that incoherently subsamples the k-space. Next, the inverse discrete Fourier transform (IDFT) is applied to obtain noise-aliased sensitivity-calibrated images. Lastly, each channel is pointwise multiplied by the conjugate of the sensitivity map for each coil, and the channels are summed to return to image space.

![](https://cdn-images-1.medium.com/max/2000/0*XOBYuUna_J3DRdYU)

As can be seen in the example below, MRI images lose a lot of their quality when the mask is applied to k-space. Our project aims to investigate methods that reconstruct the full resolution images from a subsampled k-space. By preprocessing MRI images to different initial resolutions and subsampling their respective k-space grids, we compared a Normalizing Flows model with [DeepInPy](https://github.com/utcsilab/deepinpy), a research project developed by Professor [Jon Tamir](http://users.ece.utexas.edu/~jtamir/) at The University of Texas at Austin.

![](https://cdn-images-1.medium.com/max/2000/0*z8Mc9b-FzyYmizkg)

## Data Preprocessing

Our dataset was obtained from Dr. Tamir in the form of a 50 GB HDF5 file with 4,480 knees scans, each containing the MRI image itself, 8 coil sensitivity maps, the corresponding 8 k-space grids, and a mask to incoherently subsample the k-space. This data originally came from the “Stanford Fully Sampled 3D FSE Knees” dataset on [mridata.org](http://mridata.org), containing 3D Cartesian proton-density knee scans from the Lucas Center at Stanford Medicine. The table below describes the dataset as we received it.

![](https://cdn-images-1.medium.com/max/2000/0*FZqBLhFGp2TfMRfv)

Since the full size MRI images proved to be too complicated for the Normalizing Flows model under our time and computational constraints, we experimented with preprocessing the images in different ways. The first step involved implementing the forward and adjoint sensing models to ensure the k-space, maps, and masks remained consistent with our transformed images. To apply FFT and IFFT, we used functions from this [GitHub repository](https://github.com/ismrmrd/ismrmrd-python-tools/blob/master/ismrmrdtools/transform.py). Our forward and adjoint sensing functions are shown below.

![](https://cdn-images-1.medium.com/max/2000/0*4TiSqsJFdVtyS0dt)

There are two different types of transforms we applied to the images: bit scaling and resizing. In bit scaling, the norms of the image pixels are normalized to integers between 0 and 2ⁿ-1, where n is the number of bits. Then, the images are scaled down by a factor of 2ⁿ-1, resulting in 2ⁿ discrete values between 0 and 1. This transformation makes it much easier to train the Normalizing Flows model by reducing the number of possible values for each pixel.

Because bit scaling is applied to the norm of the images, we lose information about the phase of each pixel. However, this information is not important for the purposes of our project, and the forward and adjoint sensing models are not deprecated by this change. A comparison between an original image (on the left) and the 4-bit scaled version (on the right) are shown below. The difference is barely recognizable by the human eye.

![](https://cdn-images-1.medium.com/max/2000/0*baYQDRhUuj2DF263)

Resizing, on the other hand, was a bit more complicated. Scaling the size of the images by a factor of ¼ results in 80x64 pixels. However, this transformation also needs to be applied to the maps and masks to generate the correct k-space. For the maps, the real and imaginary components had to be separated, scaled individually, and then combined again. For the masks, a different interpolation method called “inter_nearest” was used instead of “inter_area” to maintain binary values of 0 and 1. The images below compare 8-bit full scale (on the left) with 8-bit ¼ scale (on the right) for the images, maps, noise-aliased images, and masks. Resizing did not deprecate the forward and adjoint sensing models, but there was a noticeable decrease in quality for the noise-aliased images and masks.

![](https://cdn-images-1.medium.com/max/2000/0*dAh_cAaiWVLeI55j)

When creating our testing datasets, we chose 15 knees arising from different cross sections to introduce variance to our dataset. These 15 knees were processed in four different ways: 8-bit full scale, 4-bit full scale, 8-bit ¼ scale, and 4-bit ¼ scale. The ¼ scale (80x64) datasets were used to compare the Normalizing Flows model with DeepInPy since we were unable to train the Normalizing Flows model on full scale images. The full scale datasets (320x256), which DeepInPy performs much better on, were used to compare the ResNet and U-Net implementations of DeepInPy. The tables below describe the differences between the full scale and ¼ scale datasets.

![](https://cdn-images-1.medium.com/max/2000/0*zavBrFIsq9IXgB1V)

![](https://cdn-images-1.medium.com/max/2000/0*uq_RVA6qsq5rvOyK)

## Approaches

## Normalizing Flows Model

A Normalizing Flows generative model attempts to learn a complicated distribution by applying a series of invertible transforms on a simpler distribution.

![](https://cdn-images-1.medium.com/max/3200/0*o0__x854Wo4BbkIE)

[*Flow-based Deep Generative Models](https://lilianweng.github.io/lil-log/2018/10/13/flow-based-deep-generative-models.html), by Lilian Weng*

For example, our simple “base” distribution (p₀) can be a Gaussian. Then we apply the invertible transforms f₁ through fₙ to get a more complicated distribution. The goal of training the model is to find the best parameters for those transforms so that our final distribution is close to the one we attempt to model.

## Transform Functions:

There has been a lot of research on the transform functions, resulting in publication of models such as NICE and RealNVP. Common functions include affine coupling layers (ACL) as well as quadratic and cubic splines. We decided to use a more complex function called rational-quadratic splines as introduced by Durkan et al. [1]. The Neural Spline Flows model has been tested on a variety of datasets such as MNIST, CIFAR-10 and ImageNet, and has outperformed other flow models such as Glow.

## Training Loss:

The training criterion for flow models is the negative log-likelihood over the training dataset D, since the log-likelihood of the input data can be tracked. The equation below represents this relation.

![](https://cdn-images-1.medium.com/max/2000/0*WU7C6t-Y14JzahJT)

## Training:

We trained our Neural Spline Flows model on 4000 images from the MRI knees dataset obtained from Dr. Tamir, using a 3600–400 train-test split. We wrote code for loading, storing, and preprocessing the MRI dataset because the Neural Spline Flows code supported training only for datasets from the paper. In addition, the images were preprocessed to 4-bit 80x64 size as detailed earlier. The training took about 3 hours on a P100 GPU on Colab for about 70 epochs. Here are 9 of our generated knee images.

![](https://cdn-images-1.medium.com/max/2000/0*4Y5fbASd6qwpuNxV)

## Why Use a Flow Model as Prior

It turns out that having a good estimation of the prior probability enables us to complete many downstream tasks such as filling in incomplete data samples. Using a generative adversarial network (GAN) as a prior would restrict us due to its inability to represent any arbitrary input, and would therefore lead to poor reconstruction [2].

![](https://cdn-images-1.medium.com/max/2000/0*HYGqQ9L24i_V6Pyy)

The above equation represents the traditional approach to the MRI inverse problem using generative models. Here, y is our observed, undersampled k-space data, G(z) is the output of the generative model for an input z, and f is the forward sensing model defined earlier. The model output, G(z), looks similar to an MRI knee image.

Generative models have shown good results when used as priors to minimize this expression [2]. The traditional approach restricts the solution using L1 regularization to avoid overfitting. Essentially, the equation tries to minimize the difference between the undersampled k-space data and a clean image generated by the model after the function f was applied to it. The ideal solution would be to find the image x that, when passed through the forward model, gives the output y.

Another approach involved minimizing the following equation that was first implemented by Whang et al. [3].

![](https://cdn-images-1.medium.com/max/2000/0*xR277Z2vWK9lAJXK)

This approach takes advantage of the flow model’s capabilities. Since we are able to approximate the distribution of the images, we can now calculate the probability that an image came from our model’s distribution. This allows for a new kind of regularization penalty represented in the equation above. The model is still trying to produce an image that is as close to y as possible when f is applied to it, but it also tries to maximize the probability that our solution image came from the model’s distribution, meaning that the image looks like an actual MRI knee image.

However, even with a lot of tuning, the approach didn’t give good results, so we tried another approach.

![](https://cdn-images-1.medium.com/max/2050/0*DKVtcYkr_m6SWGEj)

This third approach, which is a combination of the previous two, gave better results than the others and is the first time we’ve seen it applied. To be specific, our approach performed better than simple L1 regularization even when we kept all parameters the same and just tuned beta. If we had more time, we would attempt to tune both lambda and beta to find an even better combination. However, here we just took our best tuned L1 coefficient and tuned the added log probability penalty by keeping lambda constant.

We used gradient descent methods to minimize the previous expressions over z. Specifically, we use the pytorch implementation of the Adam optimizer [4]. Getting good reconstruction results took a lot of time since we needed to tune the following parameters:

1. Optimizer learning rate

1. Number of iterations for gradient descent

1. Regularization coefficients (beta and lambda)

We implemented the inverse problem code for the flows ourselves, so our understanding of how the inverse problem is solved in practice improved. Because the k-space and map data are complex numbers, some additional preprocessing methods for the flow model were required. For example, we had to stack the real and imaginary components into two real channels. For this reason, we had to write custom functions for complex multiplication, complex conjugate, and other operations. Also, we had to write the forward and adjoint sensing transforms in a way that would work with the shape of our tensors.

Our test sets for the inverse problem consisted of 15 80x64 images in 4-bit and 8-bit format. In this table, we present the results of the flow prior reconstruction using the peak signal to noise ratio metric from scikit-image.

![](https://cdn-images-1.medium.com/max/3152/0*PF3gxk9Zr2q5VhgA)

The L1 coefficient was fine-tuned for the 4-bit reconstruction task. Then we tuned the log_probability coefficient for the 4-bit task. Both coefficients are the same for the 8-bit task. All other parameters were tuned for the 4-bit L1 regularization task and remained the same to get consistent comparison results.

It was interesting to see that our new method performed better in both test sets even without a lot of tuning. Although our model was trained on 4-bit images, it also did very well reconstructing 8-bit images. When we compare the different regularization methods, we see similar performance except for some images where our method gives much better results, shown below.

![](https://cdn-images-1.medium.com/max/2164/0*fSH3UgSpVwWoC3PC)

Here are the results for all of the images, using the same legend.

**4-bit**

![](https://cdn-images-1.medium.com/max/3200/0*N5t0W1z6eicoU13N)

**8-bit**

![](https://cdn-images-1.medium.com/max/3200/0*rsTGrLbfM4yM-aUL)

## DeepInPy

[DeepInPy](https://github.com/utcsilab/deepinpy) is a research project of Professor [Jon Tamir](http://users.ece.utexas.edu/~jtamir/) at the University of Texas at Austin to simplify creating solutions for deep inverse problems. It aims to invert forward sensing models through a combination of iterative algorithms and deep learning. Originally, DeepInPy was implemented for the specific purpose of compressed sensing MRI, so much of the code infrastructure was already available to us for this purpose prior to starting our project.

## Theory and Implementation

DeepInPy implements deep basis pursuit, which is a variation of basis pursuit denoising that utilizes deep learning. The equations for deep basis pursuit are shown below. N(x) is a CNN that estimates the aliasing of the image, and R(x) is a CNN that attempts to denoise the image. The data consistency parameter that the minimization is bound to ensures that the image is still consistent with its original k-space within a known noise power. This method allows for expected noise while finding the cleanest version of the image [5].

![](https://cdn-images-1.medium.com/max/2000/0*ItZf1I-cpmx1SO0U)

![](https://cdn-images-1.medium.com/max/2000/0*PRESL72qJOMhQlnD)

* N(x) = CNN to estimate aliasing

* R(x) = CNN to denoise x

* w = weights of the CNN

* x = image

* y = k-space

* A = forward sensing model

* ϵ = known noise power

The way deep basis pursuit is actually implemented within DeepInPy involves an iterative approach of alternating minimizations between the CNN and data consistency term. With this implementation, rₖ is the kth version of the CNN update, while xₖ is the kth version of the data consistency update, solved via ADMM. Training jointly solves the CNN weights and image reconstructions, while inference time freezes the CNN weights. During supervised learning, loss is calculated in image space by comparing the reconstructed image with the ground truth. On the other hand, unsupervised learning computes loss in k-space by comparing the forward transformed constructed image with the subsampled k-space [5].

![](https://cdn-images-1.medium.com/max/2000/0*Yptwcr9r0uaHEL4g)

![](https://cdn-images-1.medium.com/max/2000/0*ydeocm4mwrSJd6DZ)

![](https://cdn-images-1.medium.com/max/3200/0*K3eu9lpoqlgsn77N)

[*Unsupervised Deep Basis Pursuit](https://arxiv.org/pdf/1910.13110.pdf), by Tamir et al. [5]*

## Contributions to DeepInPy

### Documentation

In the course of the project, we also decided to help resolve one of DeepInPy’s GitHub issues by creating the documentation necessary to quickly understand the structure of DeepInPy and how it can be used. We started by simply working with the code and trying to understand it ourselves through application; once this was complete, we moved onto writing docstrings on a per-file level.

These docstrings are predominantly centered around illuminating the System-Model-Optimization (SMO) block model that DeepInPy uses; the System block includes datasets and forwards, the Model block includes deep learning implementations such as a ResNet, and the Optimization block includes gradient algorithms like conjugate gradient descent. This SMO structure is encapsulated within the Recon object, an abstract class built for deep learning, which is implemented by users to create the actual deep inverse model architectures.

After completion of the docstrings, they were peer-reviewed within the group before being submitted within a forked DeepInPy repository for Dr. Tamir’s approval.

### U-Net

The current official implementation of DeepInPy utilizes the mentioned System-Model-Optimization block model for the Recon object, which transforms the data y to a reconstruction variable x_hat with the forward model A and parameters θ. The Recon object is structured such that one can modify, add, or remove individual blocks, such as adding a new model like U-Net to the Loop Block or replacing the Conjugate Gradient Block with another Optimization Block. For our purposes, we sought to see if we could outperform the established Model Block ResNet with another Model Block based on U-Net.

![](https://cdn-images-1.medium.com/max/2412/0*V9nnwXPVi69I9lXw)

[*Current Recon Implementation](https://github.com/utcsilab/deepinpy), by Jon Tamir*

U-Net is designed to solve the image segmentation problem but has also been used for the image denoising problem [6], which makes it an effective substitute for ResNet to reduce noise within the image. We made a derivative implementation of the original U-Net published by a group from the University of Freiburg [7], which originally consisted of a contracting path of convolutional layers and an expanding path of deconvolutional layers, where the contracting path is used to obtain the context of the image, and the expanding path keeps the localization of the image itself [6].

![](https://cdn-images-1.medium.com/max/3110/0*KyFi7JjkpReiH5OQ)

[*Original U-Net implementation](https://arxiv.org/abs/1505.04597), by Olaf Ronneberger*

For our implementation of U-Net, we started with a modified version of the U-Net from the original paper [6], found in this [GitHub repository](https://github.com/n0obcoder/UNet-based-Denoising-Autoencoder-In-PyTorch/blob/master/unet.py), with some minor changes from Komatsu’s implementation for denoising RGB images [7], as well as a select number of implementation differences to apply it to DeepInPy. We maintain the relative architecture of the U-Net, but originally expanded to a size of 320x256x32 for the full scale images and 80x64x32 for the 1/4th scale images. We also introduce zero-padding to the image to maintain the same input and output size, to allow for compatibility with the general recon architecture of DeepInPy. In the expanding phase, instead of the up-convolution, we upsampled by a factor of 2 and performed a 1x1 convolution, halving the number of output channels in the process. Due to GPU resource limitations, we limited the overall depth of the U-Net to be 3 layers, compared to the 5 layers presented in the paper, such that the maximum depth of the network is 128 channels. The number of input channels was modified to accept 2 channels to accommodate the complex inputs.

## Results

## ResNet and U-Net Comparison

With comparison to ResNet, the U-Net block resulted in limited successes. For comparison, we ran both models with the 8-bit full scale and 4-bit full scale datasets and compared the results.

![](https://cdn-images-1.medium.com/max/2000/0*MrBzysPuPX_3dgY0)

![](https://cdn-images-1.medium.com/max/2000/0*z_jQgbASJ64n_iRL)

For both ResNet and U-Net, the models were trained for 500 epochs on the 8-bit data and 1000 epochs on the 4-bit data. Both were performing unsupervised learning with the noisy image as input. They were run with 4 unrolls of the loop block, where the ResNet or U-Net block was followed by a conjugate gradient optimization block. Both models were run with a batch size of 5. The learning rate was optimized to converge for both models while being maximized. For ResNet this was 0.00004, and for U-Net this was 0.0011.

For the 8-bit data, it is visually clear that there is a performance loss switching the CNN model used from ResNet to U-Net. However, the difference for the 4-bit data is less apparent. We also calculated the normalized root mean square loss to the ground truth for each model. In both datasets, the final NRMSE for U-Net was approximately 0.18. For ResNet, the final NRMSE for the 8-bit was approximately 0.13, and for the 4-bit it was approximately 0.16.

It is important to note that the NRMSE values across the two datasets are not directly comparable without accounting for the difference in the number of Epochs, which was due to resource limitations and time constraints. However, it is evident that the difference in the outputs of the two models was minimized by going from 8-bit to 4-bit. It is possible, observing the trend lines for the 4-bit outputs, that there could be significant further gains with an increase in the number of Epochs for U-Net while less so for ResNet, potentially decreasing the performance difference between the two models.

![](https://cdn-images-1.medium.com/max/2000/0*NXD_qxmsZm7aUO-r)

## Normalizing Flows and DeepInPy Comparison

We compared the DeepInPy and Normalizing Flows reconstruction on the 80x64 4-bit and 8-bit datasets.

![](https://cdn-images-1.medium.com/max/2126/0*2_fNlpmMQc4-eiXa)

![](https://cdn-images-1.medium.com/max/2126/0*FZ5gRjummo4y_hAu)

Overall the results from DeepInPy are more consistent; however, the Normalizing Flows model gives more accurate results in certain cases. Comparing the NRMSE between the two, we see that the Normalizing Flows model outperforms DeepInPy in the case of the quarter scaled images. Both models can be easily improved by utilizing more data and GPU computational power to efficiently process the data and train the models, as computational power was the main limitation for training the models.

![](https://cdn-images-1.medium.com/max/3160/0*nVna-nV9ndl6c11o)

## Conclusion

This project aims to bring together three different scientific areas of MRI, inverse problems, and deep learning. Specifically, we applied current research methods and devised new ones to help solve compressed sensing MRI. During the process, we had the opportunity to contribute to an active research project, namely DeepInPy led by Professor Jon Tamir. We added our own version of the U-Net autoencoder as an alternative to the existing ResNet model, along with necessary documentation for others to utilize DeepInPy.

In addition, we solved the inverse problem by using a trained Normalizing Flows model as the prior. To the best of our knowledge, this is the first time an MRI inverse problem has been approached in such a way. To achieve this, we wrote our own inverse problem code and came up with our own regularization methods that outperformed some more traditional ones. In general, we got very good reconstruction results for our images with both the DeepInPy and the Flow approaches. It was a tremendous learning experience for the team and has laid the foundations for similar research work in the future. *Most importantly, we had a wonderful time while working on it.*

## References

Our [GitHub Repository](https://github.com/SamiKhandker/ImprovingMRIQuality) (most of our code is in the form of Colab Notebooks)

[1] [[1906.04032] Neural Spline Flows](https://arxiv.org/abs/1906.04032) (Durkan et al.)

[2] [[1905.11672] Invertible generative models for inverse problems: mitigating representation error and dataset bias](https://arxiv.org/abs/1905.11672) (Asim et al.)

[3] [[2003.08089] Compressed Sensing with Invertible Generative Models and Dependent Noise](https://arxiv.org/abs/2003.08089) (Whang et al.)

[4] [[1412.6980] Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980) (Kingma and Ba)

[5] [[1910.13110] Unsupervised Deep Basis Pursuit: Learning inverse problems without ground-truth data](https://arxiv.org/abs/1910.13110) (Tamir et al.)

[6] [[1505.04597] U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597) (Ronneberger et al.)

[7] [Effectiveness of U-Net in Denoising RGB Images](https://aircconline.com/csit/abstract/v9n2/csit90201.html) (Komatsu and Gonsalves)
