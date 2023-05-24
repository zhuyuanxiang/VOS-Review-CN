# Section 4. 方法

## Section 4.4. 基于光流的VOS方法

前面几节：Sect.4.1、Sect.4.2和Sect.4.3介绍了VOS方法中经常使用的空间技术。这些技术有利于解决VOS中的几个挑战，如：遮挡、视野外和快速运动。然而，在分割具有外观变化、尺度变化或者仅考虑空间特征不能生成高质量结果。因此，一些专注于连续视频帧的技术（即光流、掩膜传播和长时时间模型）已经被提出来解决这些挑战。

因为存在像素级运动模式，光流法一直是VOS中广泛使用的技术。该技术假定目标对象和背景拥有不同的运动模式。因此，将光流纳入VOS可以提供具有合理先验的分割网络，如图16所示。早期的方法将估计的光流图明确地融合到他们的分割网络。为了进一步将时间线索隐匿地编码进光流图中，最近一些方法采用光流来建立帧之间的短时对应关系。图17描述了代表性的工作。在本了中，Sect.4.4.1 首先讨论了早期的工作，Sect.4.4.2 然后介绍了最近的扩展，Sect.4.4.3 总结讨论的方法。表12简要地列出了这些方法的特点。

表12. 基于讨论过的基于光流的VOS方法的汇总。$(t-1,t)$从$I_{t-1}$到$I_t$测量光流图。“计算和使用”说明了所列的方法如何计算和使用光流图。

| Years | Venues | Types | Paper titles with links (abbreviates in our review)                                                                                                                                                                                     |                                            Codes                                            | 光流图                            | 计算和使用                           |
| :---: | :----: | :---: | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :-----------------------------------------------------------------------------------------: | --------------------------------- | ------------------------------------ |
| 2017  |  CVPR  |   U   | [Learning Motion Patterns in Videos](https://openaccess.thecvf.com/content_cvpr_2017/papers/Tokmakov_Learning_Motion_Patterns_CVPR_2017_paper.pdf) (MP-Net)                                                                             |                     [Website](http://lear.inrialpes.fr/research/mpnet/)                     | $(t-1,t)$                         | LDOF；生成初始对象掩膜               |
| 2017  |  CVPR  |   U   | [FusionSeg: Learning to combine motion and appearance for fully automatic segmentation of generic objects in videos](https://openaccess.thecvf.com/content_cvpr_2017/papers/Jain_FusionSeg_Learning_to_CVPR_2017_paper.pdf) (FusionSeg) |                     [Caffe](https://github.com/suyogduttjain/fusionseg)                     | $(t-1,t)$                         | CNN模块；与外观融合（预测时）        |
| 2017  |  ICCV  |  S&U  | [SegFlow: Joint Learning for Video Object Segmentation and Optical Flow](https://openaccess.thecvf.com/content_ICCV_2017/papers/Cheng_SegFlow_Joint_Learning_ICCV_2017_paper.pdf) (SegFlow)                                             |                      [Caffe](https://github.com/JingchunCheng/SegFlow)                      | $(t-1,t)$                         | CNN模块；与外观融合（预测时）        |
| 2018  |  CVPR  |   S   | [CNN in MRF: Video Object Segmentation via Inference in A CNN-Based Higher-Order Spatio-Temporal MRF](https://openaccess.thecvf.com/content_cvpr_2018/papers/Bao_CNN_in_MRF_CVPR_2018_paper.pdf) (CINM)                                 |                                                                                             | $(t-2,t),(t-1,t),(t,t+1),(t,t+2)$ | FlowNet2.0；建立像素间的时间依赖关系 |
| 2018  |  CVPR  |   S   | [MoNet: Deep Motion Exploitation for Video Object Segmentation](https://openaccess.thecvf.com/content_cvpr_2018/papers/Xiao_MoNet_Deep_Motion_CVPR_2018_paper.pdf) (MoNet)                                                              |                                                                                             | $(t-1,t)$                         | FlowNet2.0；背景抑制                 |
| 2018  |  ECCV  |   U   | [Unsupervised Video Object Segmentation with Motion-based Bilateral Networks](https://openaccess.thecvf.com/content_ECCV_2018/papers/Siyang_Li_Unsupervised_Video_Object_ECCV_2018_paper.pdf) (MBN-VOS)                                 | [Tensorflow](https://github.com/siyangl/video-object-segmentation-motion-bilateral-network) | $(t-1,t);(t,t+1)$                 | FlowNet2.0；特征包装和背景抑制       |

### 4.4.1 早期的基于光流的方法

MP-Net（motion Pattern Network）[^Tokmakov,2017a]是最早使用光流的深度方法之一。LDOF（Large Displacement Optical Flow）[^Brox&Malik,2010a]被用来生成帧之间的光流图，这些光流图通过深度网络精调，然后与对象可能性得分融合，得出最终的结果。讨论：由于全面的运动模式和互补的信息融合，MP-Net相比传统的基于运动的VOS方法可以更好的从复杂场景中分离移动的对象。

FusionSeg[^Jain,2017]实现了一个并行机制来融合运动特征和语义信息。与MP-Net不同的是，FusionSeg将基于运动的结果与对象的可能性执行后处理，然后将基于运动和基于外观的分离结果用两个独立的分支进行编码，并且之后将其合并。讨论：因此，组合模块可以被训练成同时地利用两种信息，并且生成更多精确的结果。

SegFlow[^Cheng,2017]也是一个融合运动和外观特征的并行网络。与FusionSeg不同，SegFlow为不同的目标训练两个分支（分别用于分割和光流）。因此，该方法不同直接混合两个分支。相反，这种结合是在上采样阶段实现的。讨论：相比MP-Net和FusionSeg，SegFlow在运动特征和外观特征构建了一个更加紧致的桥梁，从而为UVOS和SVOS提供了一个端到端的可训练架构。

### 4.4.2 基于光流方法的扩展

光流为目标对象带来了像素级的定位和形状先验，使得VOS方法在一些基准数据集上产生了有竞争力的结果。然而，受限于训练数据的缺乏和挑战（移动的背景和相机的晃动），估计的光流图的质量并不能总是提供这种有价值的先验。因此，好几种方法被提出用于从光流图中挖掘出更加可靠的时间信息。

CINM（Cnn IN Mrf）[^Bao,2018]通过最小化MRF模型来执行SVOS：
$$
E(\mathbf{x})=\sum_{i\in\mathcal{S}}E_u(x_i)+\sum_{(i,j)\in\mathcal{N}_r}E_t(x_i,x_j)+\sum_{c\in\mathcal{S}}E_s(\mathbf{x}_c)
$$
其中$E_u,E_t,E_s$分别表示似然性最大化、时间依赖和空间依赖的能量。$x$描述了由OSVOS预测的初始标签，$\mathcal{V}$定义了视频序列中的所有像素。局部连接$\mathcal{N}_T$通过光流建立。$\mathbf{x}_c$是第$c$帧的辅助掩膜，通过$x$进行初始化，通过基于DeepLab v2的模块进行精调。讨论：与上述方法不同的是CINM将光流图视为MRF中的一个约束项，而不是明确使用它们来生成掩膜。虽然取得了有竞争力的结果，但是CINM的计算成本是巨大的，因为它融合了几个深度网络（OSVOS、FlowNet和DeepLab）。

MBN-VOS（Motion-based Bilateral Network for UVOS）[^Li,2018c]实现了一个双向网络（Bilateral Network）[^Jampani,2016]，这个双向网络基于对象可能性和光流生成了背景运动模式。产生的模式可以作为先验以减少静态背景对象对最终结果的负面影响。讨论：与CINM不同，在MBN-VOS中的光流主要用于背景抑制，从而提升分割性能，特别是当背景对象看起来与前景对象相似时。

MoNet[^Xiao,2018]实现了类似于MBN-VOS的背景抑制方法。然而不同的是，MoNet仅从光流分支中生成了背景运动模式。此外，前向与反向的光流都被估计为包装为邻近的帧特征。讨论：通过双向光流图和包装的特征，MoNet可以有效地编码帧之间的时空对应，改善VOS性能。

### 4.4.3 基于光流的VOS方法汇总

本节讨论了具有代表性的基于光流的VOS方法。这些方法假定目标对象和背景拥有不同的运动模式。因此，生成的深度图可以合理地估计目标对象的形状和位置先验。早期的方法（MP-Net、FusionSeg、SegFlow）显式地将光流图与空间特征结合起来，以生成对象掩膜。它们之间的主要区别是如何编码和结合不同类型的特征。然而，由于缺乏训练数据和具有挑战性的序列（如：动态背景），导致估计的光流并不总是可靠的。因此，最近提出了几种方法来更加充分地利用光流，同时避免上述风险。如上所述，CINM基于光流的时间依赖性隐含地约束了VOS的结果；MBN-VOS和MoNet通过识别背景运动模式抑制了背景区域。

虽然在许多具有挑战性的序列上取得了高质量的结果，但是光流在最近的VOS系统中很少采用，因为：⑴在某些情况下，对象和背景光流图是没有区别的（如：静态对象）；⑵在推理过程　需要额外的深度网络（如：FlowNet，详见表3、表4、表5）。因此，这种技术已经逐渐被掩膜传播所取代，这将在下一节讨论。
