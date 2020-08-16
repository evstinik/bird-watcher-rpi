#!/bin/bash

echo "WARN: Run this only after upload_client.sh"

scp ./config/bird-watcher-client.service pi@192.168.0.186:/home/pi
ssh pi@192.168.0.186 "cd /lib/systemd/system && sudo mv ~/bird-watcher-client.service . && sudo chmod 644 bird-watcher-client.service && sudo systemctl daemon-reload && sudo systemctl enable bird-watcher-client.service"
