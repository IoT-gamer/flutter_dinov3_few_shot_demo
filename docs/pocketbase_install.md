# PocketBase Local Installation Guide

## Install unzip if you don't have it already
```bash
sudo apt install -y unzip
```
## Download the latest PocketBase release (replace the URL with the latest version if needed)

```bash
wget https://github.com/pocketbase/pocketbase/releases/download/v0.36.1/pocketbase_0.36.1_linux_amd64.zip
```
## Create a directory for your application and unzip the file into it
```bash
mkdir pb
unzip pocketbase_0.36.1_linux_amd64.zip -d pb
```
## Navigate to the directory and make the binary executable
```bash
cd pb
chmod +x pocketbase
```

## Run PocketBase and Bind to Your Network IP 
- By default, PocketBase may only bind to the loopback address (127.0.0.1), making it inaccessible from other devices on your local network. To access it from other devices, you need to bind it to your local network's IP address or 0.0.0.0 (all interfaces). 

```bash
./pocketbase serve --http="0.0.0.0:8090"
```

By default, PocketBase will run on port 8090. You can access the admin dashboard at `http://localhost:8090/_/`.

## Check UFW firewall status
```bash
sudo ufw status
```
If UFW is active, Allow the default PocketBase port (TCP port 8090):
### from local network only:
```bash
sudo ufw allow from 192.168.1.0/24 to any port 8090 proto tcp
```