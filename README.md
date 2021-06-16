# KubeAsynDGAN

This repository features an implementation for a reproduction for the paper "Synthetic Learning: Learn From Distributed Asynchronized Discriminator GAN Without Sharing Medical Image Data" by Chang et al. The implementation features a simplified neural network and is trained on the MNIST dataset. The network is trained, in a distributed way, on the KubeML network created by Diego Albo. Before running the implementation please make sure KubeML and its components. Go to https://github.com/DiegoStock12/kubeml/ to see how one can set up KubeML. One can also run the `cluster_config.sh` script below to set it up. 

* Install the KubeML CLI with `curl -Lo kubeml https://github.com/diegostock12/kubeml/releases/download/0.1.2/kubeml && chmod +x kubeml`

* Run `sh cluster_config.sh` on a GKE cloud shell to set up the rest of system

### Running the KubeAsynDGAN training script

The KubeAsynDGAN training script takes care of everything, but it needs the discriminator function to be enlisted in KubeML.

* run `./kubeml fn create --name disc-load --code function_discriminator_load.py` before running the training script

The training script is `run.py`. First it opens a port to the RedisAI pod to load and enlist models, starting the script it also initializes the latest Discriminator model. It generates the dataset with the current Generator and enlists it in KubeML automatically, after deleting the old dataset. 
Running the first time the MNIST dataset also needs to be downloaded, this means that in `train_generator.py` one should set `download=True` in the dataset loader. After doing this one time, one can set it to false to prohibit it downloading the dataset every epoch (which is unnecessary). 

Then the script trains the discriminator on the Kubernetes cluster with KubeML asynchronously and checks when the script has finished, in the code one can alter the hyperparameters. When the discriminator training is finished, the latest discriminator model is pulled from the RedisAI server to train the Generator locally. 
After this main epoch, the generator model is saved in the `models` folder, which can later be used to generate images. However one can also take the generated `.npy` datasets and pull the synthetic dataset from it. The dimensions are (10000, 1, 2, 28, 28) where the dimension with `2` denotes the dimension for "fake" and "real" images. 

* `python3 run.py` to run the system after the network is added to the KubeML functions. Parameters for training need to be set in code on the function calls!

The training script invokes `generate_dataset_train.py` and `train_generator.py`.


Finally, `run_async.py` is an old training file not loading model weights but showing a more asynchronous version of the training. There is also a notebook that shows some exploration of dataset creation and a `.txt` file with some KubeML and Kubernetes commands that help in debugging.
