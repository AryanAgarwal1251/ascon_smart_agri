#!/bin/sh
# One-shot: demo channel keys for the sealed weight exchange and a self-signed demo TLS
# certificate for the cloud receiver, into shared volumes. Never leaves the compose project.
set -eu
if [ ! -f /keys/phase8.demo.key ]; then
  asa aggregator --generate-keys --keys /keys/phase8.demo.key --clients pi-1,pi-2,sim-3
fi
if [ ! -f /certs/cloud.demo.pem ]; then
  asa cloud-receiver --generate-cert --cert /certs/cloud.demo.pem --key /certs/cloud.demo.key \
      --cert-hostname cloud-receiver
fi
echo "[init] demo keys + cert ready"
