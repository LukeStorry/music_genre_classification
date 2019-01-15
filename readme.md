# Music Genre Classification with Tensorflow

Replicating parts of Schindler's [paper](https://publik.tuwien.ac.at/files/publik_256008.pdf)
on Comparing Shallow versus Deep Neural Network Architectures for Automatic Music Genre Classification.

For COMSM0018.

### Usage
```python main.py [options]```

command line options are:
  - ```--depth [shallow or deep]``` (default shallow)
  - ```--epochs [int]``` (default 100)
  - ```--samples [int]``` (default 11250)
  - ```--augment [bool]``` (default False)
  - ```--batch_normalisation [bool]``` (default False)
  -  ```--batch_size [int]``` (default 16)

For example, to run a deep network with data-augmentation, with only 20 epochs and 1000 audio samples:
```python main.py --depth deep --augment True --epochs 20 --samples 1000```


### Requirements
  - Tensorflow 1.6
    - (The following is required for running on Blue Crystal)\
    ```module add languages/anaconda2/5.0.1.tensorflow-1.6.0```
  - librosa
