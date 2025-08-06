# CNN-GRU: A novel vanadium redox flow battery state of charge estimation method utilizing convolutional neural network and gated recurrent unit for different operating conditions

Code release for our Journal of Power Sources paper 'A novel vanadium redox flow battery state of charge estimation method utilizing convolutional neural network and gated recurrent unit for different operating conditions'

Runtian Li, Yue Wang, Tianyi Zhang

## Introduction

Accurate state of charge (SOC) estimation is crucial to improving the efficiency of vanadium redox flow battery (VRFB) and extending their service life. However, previous methods generally rely on prior electrochemical knowledge, and do not fully consider the flow rate that significantly affects electrochemical behavior. This paper proposes a novel CNN-GRU hybrid model that combines convolutional neural networks (CNN) with gated recurrent units (GRU). It leverages the complex feature extraction capability of CNN to capture key information from test data such as voltage, current, and flow rate. Subsequently, GRU establishes a nonlinear mapping relationship to achieve accurate and stable SOC estimation, relying on its powerful temporal processing capability. This method accounts for the impact of dynamic flow rate variations and eliminates the need to build a battery model. Training and validation of the model use test data collected under varying flow rates and operating conditions. Comparisons with other methods demonstrate significant superiority, achieving an average RMSE of 2.79%, MAE of 2.28%, MAXE of 7.5%, and 𝑅2 of 98.52%. Additionally, results show that once trained, the model can achieve SOC prediction at approximately 47μs per sample with high computational efficiency, making it particularly suitable for battery management systems (BMS).


## Installation
Install requirements

```sh
pip install -r requirements.txt
```

## Training and Testing

```
python3.11 __main__.py
```

## Citation
If you find this code useful, please consider citing: 

```bibtex
@ARTICLE{10382587,
  author={Runtian Li, Yue Wang, Tianyi Zhang},
  journal={Journal of Power Sources}, 
  title={A novel vanadium redox flow battery state of charge estimation method utilizing convolutional neural network and gated recurrent unit for different operating conditions}, 
  year={2025},
  volume={},
  number={},
  keywords={Vanadium redox flow battery; State-of-Charge; Hybrid data-driven model; Flow rate; Convolutional neural network; Gate recurrent unit},  
  doi={}
}
```
