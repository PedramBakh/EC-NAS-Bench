#!/bin/bash
for pcidev in $(lspci -D|grep 'VGA compatible controller: NVIDIA'|sed -e 's/[[:space:]].*//'); do echo 0 > /sys/bus/pci/devices/${pcidev}/numa_node; done
