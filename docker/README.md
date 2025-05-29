# Docker Tutorial for [DIAL-MPC](https://github.com/LeCAR-Lab/dial-mpc)

This folder contains the Dockerfile of the DIAL-MPC.
DIAL-MPC is an impressive framework for legged robot ***full-order torque-level*** control with both precision and agility in a ***training-free*** manner.
You can find the DIAL-MPC in [here](https://github.com/LeCAR-Lab/dial-mpc).

## Requirenments
> [!CAUTION]
> You need a Linux or Ubuntu system with Nvidia-driver (>=525.60.13) to support CUDA 12.*

## Docker and Nvidia-toolkit installation
### Step 1. Docker install
``` bash
curl -fsSL https://get.docker.com -o get-docker.sh
sudo bash get-docker.sh
``` 
### Step 2. Docker Postinstall(Follow [here](https://docs.docker.com/engine/install/linux-postinstall/))
### Step 3. Docker Nvidia Toolkit(Follow [here](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html))

### Step 4. (Optional) Install xhost
``` bash
sudo apt install x11-xserver-utils
```
## Build Docker Env
``` bash
git clone https://github.com/XuXinhangNTU/dial-mpc-docker
cd dial-mpc-docker
docker build -t dial-mpc .
```
### First Time Run (Create the container)
```bash
xhost +
docker run -it -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=unix$DISPLAY --gpus all --name <Your Name> dial-mpc
```
### Second Time Run
```bash
xhost +
docker start <Your Name> && docker exec -it <Your Name> bash
```
## Tips
The vscode plugin "Dev Containers" is very easy to use when you do docker development.

## Notice
> [!CAUTION]
> If the demo looks odd, check your GPU usage! RTX 3090 is tested.
