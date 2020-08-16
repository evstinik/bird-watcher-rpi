#!/bin/bash

cd /lib/systemd/system
sudo mv ~/bird-watcher-client.service . 
sudo chmod 644 bird-watcher-client.service
sudo systemctl daemon-reload
sudo systemctl enable bird-watcher-client.service
