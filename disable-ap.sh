#!/bin/bash
sudo systemctl stop hostapd
sudo systemctl stop dnsmasq
sudo dhclient wlan0
sudo systemctl start NetworkManager.service
echo "Wifi mode enabled"