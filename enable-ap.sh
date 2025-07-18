#!/bin/bash
sudo systemctl stop NetworkManager.service
sudo systemctl start hostapd
sudo systemctl start dnsmasq
sudo ifconfig wlan0 192.168.4.1
echo "Access Point enabled"