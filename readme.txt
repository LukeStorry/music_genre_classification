Deep Learning Coursework

Written using Tensorflow 1.6, so the following module is required for Blue Crystal:
module add languages/anaconda2/5.0.1.tensorflow-1.6.0

files required in folder:
main.py
utils.py
shallownn.py
deepnn.py


to run:
python main.py [options]

command line options are:
--depth [shallow or deep] (default shallow)
--epochs [int] (defaault 100)
--samples [int] (default 11250)
--augment [bool] (default False)
--batch_size [int] (default 16)

For example, to run a deep network with data-augmentation, with only 20 epochs and 1000 audio samples:
python main.py --depth deep --augment True --epochs 20 --samples 1000
