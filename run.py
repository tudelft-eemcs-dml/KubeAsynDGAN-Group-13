import subprocess, os
from train_generator import TrainGenerator
import re
import threading
import time
my_env = os.environ.copy()
my_env["KUBECONFIG"] = os.path.expanduser(f"~/.kube/config")
main_epochs = 5
disc_epochs = 1
gen_epochs = 1
refresh_dataset=True

def port_forward(pod):
    subprocess.run("kubectl -n kubeml port-forward " + redispod + " 6379:6379", env=my_env, shell=True)

# port forward to redisAI
out = subprocess.check_output("kubectl -n kubeml get pods", env=my_env, shell=True)
redispod = re.search('redis-([^\s]+)', out.decode("utf-8")).group(0)

print("Port forwarding to pod" + redispod)
th = threading.Thread(target=port_forward, args=(redispod,))
th.start()
print("sleep 5 seconds to get the stuff running")
time.sleep(5)

for i in range(main_epochs):
    print("===== MAIN EPOCH " + str(i + 1) + " =====")
    # # Generate Dataset
    print("=== Generating new dataset from updated generator ===")
    out = subprocess.check_output("python3 generate_dataset_train.py", shell=True)
    # print(out)
    print("Dataset Done")

    # Enlist dataset KubeML
    if refresh_dataset:
        print("=== Removing old dataset from KubeML ===")
        out = subprocess.check_output("./kubeml dataset delete --name mnist_gan", shell=True, env=my_env)
        print(out.decode("utf-8"))
        print("=== Enlisting new dataset in KubeML ===")
        out = subprocess.check_output('./kubeml dataset create --name mnist_gan --traindata x_train_disc.npy --trainlabels y_train_disc.npy --testdata x_test_disc.npy --testlabels y_test_disc.npy', shell=True, env=my_env)
        print(out.decode("utf-8"))

    
    # Train Discriminator with KubeML
    print("=== Training Discriminator on KubeML ===")
    out = subprocess.check_output("./kubeml train --function discriminator --dataset mnist_gan --epochs 10 --lr 0.0002 --batch 64 --parallelism 7 --static", env=my_env, shell=True)
    out = out.decode("utf-8")
    job_id = re.sub(r"\W", "", out)
    # TODO: fix newline part
    if len(job_id) < 10:
        print("job ID: " + job_id)
    else:
        print(job_id)
        print("Could not find job_id")
        continue

    # This could go after we have fixed the discriminator update!
    while True:
        print("Checking if KubeML Discriminator training job is finished...")
        time.sleep(5)
        out = subprocess.check_output("./kubeml task list", env=my_env, shell=True).decode("utf-8")
        if job_id not in out:
            print("job finished")
            break

    # Train Generator locally
    print("=== Train Generator Locally with new Discriminator Model===")
    generator_trainer = TrainGenerator(job_id=job_id)
    generator_trainer.train()
