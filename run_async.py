import subprocess, os
from train_generator import TrainGenerator
import re
import threading
import time
my_env = os.environ.copy()
my_env["KUBECONFIG"] = os.path.expanduser(f"~/.kube/config")
main_epochs = 10
disc_epochs = 1
gen_epochs = 1
refresh_dataset=True

def port_forward(pod):
    subprocess.run("kubectl -n kubeml port-forward " + redispod + " 6379:6379", env=my_env, shell=True)

# port forward to redisAI
out = subprocess.check_output("kubectl -n kubeml get pods", env=my_env, shell=True)
redispod = re.search('redis-([^\s]+)', out.decode("utf-8")).group(0)

print("=====================================")
print("=== KubeASynDGAN TRAINING PROCESS ===")
print("=====================================")
print("- Port forwarding to pod" + redispod)
th = threading.Thread(target=port_forward, args=(redispod,))
th.start()
print("- Sleep 5 seconds to get the stuff running")
time.sleep(5)
job_id = None

for i in range(main_epochs):
    print("===== EPOCH " + str(i + 1) + " =====")
    # # Generate Dataset
    print("--> Generating new dataset from updated generator")
    out = subprocess.check_output("python3 generate_dataset_train.py", shell=True)
    print("- Generated")

    # Enlist dataset KubeML
    if refresh_dataset:
        print("--> Removing old dataset from KubeML")
        out = subprocess.check_output("./kubeml dataset delete --name mnist_gan", shell=True, env=my_env)
        print(out.decode("utf-8"))
        print("--> Enlisting new dataset in KubeML")
        out = subprocess.check_output('./kubeml dataset create --name mnist_gan --traindata x_train_disc.npy --trainlabels y_train_disc.npy --testdata x_test_disc.npy --testlabels y_test_disc.npy', shell=True, env=my_env)
        print(out.decode("utf-8"))

    
    # Start Discriminator training with KubeML
    if job_id is None:
        print("--> Starting Training Discriminator on KubeML")
        out = subprocess.check_output("./kubeml train --function discriminator --dataset mnist_gan --epochs 1000 --lr 0.0002 --batch 64 --parallelism 7 --static", env=my_env, shell=True)
        out = out.decode("utf-8")
        job_id = re.sub(r"\W", "", out)
        # TODO: fix newline part
        if len(job_id) < 10:
            print("- job ID: " + job_id)
        else:
            print(job_id)
            print("- Could not find job_id, skipping Epoch")
            continue

    # Check if New Epoch is Done:
    print("--> Checking if Discriminator 10 Epochs are Done.", end='', flush=True)
    while True:
        time.sleep(5)
        out = subprocess.check_output("./kubeml logs --id " + job_id, env=my_env, shell=True)
        out = out.decode('utf-8')
        match_string = "{\"epoch\": " + str(10*(i + 1)) + "}"
        if match_string in out:
            break
        print('.', end='', flush=True)
    print("")
    # print(out)

    # Train Generator locally
    print("--> Train Generator Locally with new Discriminator Model")
    # TODO: Set batch_size and set dataset size, set run epoch
    # TODO: Set Loss in File
    generator_trainer = TrainGenerator(job_id=job_id)
   loss = generator_trainer.train()
    print("- Generator epoch done")

print("=====================================")
print("=============== DONE ================")
print("=====================================")
print("- Press ctrl + c to stop the program")
