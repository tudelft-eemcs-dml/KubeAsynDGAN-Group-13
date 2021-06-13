import subprocess, os
my_env = os.environ.copy()
my_env["KUBECONFIG"] = os.path.expanduser(f"~/.kube/config")
main_epochs = 1
disc_epochs = 1
gen_epochs = 1
refresh_dataset=False

for i in range(main_epochs):
    # Generate Dataset
    print("Generating Dataset")
    # out = subprocess.check_output("python3 generate_dataset_train.py", shell=True)
    # print(out)
    print("Dataset Done")

    # Enlist dataset KubeML
    if refresh_dataset:
        print("=== Removing old dataset ===")
        out = subprocess.check_output("./kubeml dataset delete --name mnist_gan", shell=True, env=my_env)
        print(out.decode("utf-8"))
        print("=== Enlisting new dataset ===")
        out = subprocess.check_output('./kubeml dataset create --name mnist_gan --traindata x_train_disc.npy --trainlabels y_train_disc.npy --testdata x_test_disc.npy --testlabels y_test_disc.npy', shell=True, env=my_env)
        print(out.decode("utf-8"))

    
    # Train Discriminator with KubeML
    print("=== Training Discriminator on KubeML ===")
    out = subprocess.check_output("./kubeml train --function discriminator --dataset mnist_gan --epochs 100 --lr 0.002 --batch 64 --parallelism 3 --static", env=my_env, shell=True)
    job_id = out.decode("utf-8")
    # TODO: fix newline part
    if len(job_id) < 10:
        print("=== JOB ID ===")
        print(job_id)
    else:
        print(job_id)
        print("Could not find job_id")

    job_id = "71e87af8"

    # Train Generator locally
    print("=== Training Generator Locally ===")
    # python3 train_generator.py --id job_id

    # latest job 71e87af8