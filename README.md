# KubeAsynDGAN

Go to https://github.com/DiegoStock12/kubeml/ to see how one can set up KubeML

* Run sh cluster_config.sh on a GKE cloud shell to set up the system

* run `./kubeml fn create --name disc-load --code function_discriminator_load.py` before running the training script

* `python3 run.py` to run the system after the network is added to the KubeML functions. Parameters for training need to be set in code.

The training script invokes `generate_dataset_train.py` and `train_generator.py`
`run_async.py` is an old training file not loading model weights but showing a more asynchronous version of the training. 

Finally there is a notebook that shows some exploration of dataset creation and a `.txt` file with some KubeML and Kubernetes commands that help in debugging.
