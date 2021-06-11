epochs = 5

for i in range(epochs):
    # Generate Dataset
    # python3 generate_dataset_train.py --model latest

    # Enlist dataset KubeML
    # ./kubeml dataset create --name mnist_gan --traindata x_real.npy --trainlabels x_fake.npy --testdata x_real_test.npy --testlabels x_fake_test.npy

    # Train Discriminator with KubeML
    # ./kubeml train --function discriminator --dataset mnist_gan \--epochs 10 --lr 0.01 --batch 64 \--parallelism 4 \--static
    # Response back job id

    # Train Generator locally
    # python3 train_generator.py
    None