#!/usr/bin/env bash
# Rent a "standard laptop" class Windows VM on Azure for a couple of hours of
# app testing, run the headless speech-bug probes on it, and tear it down.
#
#   ./scripts/azure-wintest.sh up      create RG + VM (D4s_v5: 4 vCPU, 16 GB)
#   ./scripts/azure-wintest.sh test    run scripts/azure-wintest-remote.ps1 on the VM
#   ./scripts/azure-wintest.sh rdp     write a .rdp file and open it (needs Windows App)
#   ./scripts/azure-wintest.sh down    delete the whole resource group
#
# Cost guardrails: auto-shutdown at 22:00 UTC daily; `down` removes everything.
set -euo pipefail

RG="${RG:-tinyexplorer-wintest}"
LOC="${LOC:-uksouth}"
VM="${VM:-wintest01}"
SIZE="${SIZE:-Standard_D4s_v5}"          # 4 vCPU / 16 GB — typical laptop class
ADMIN_USER="${ADMIN_USER:-babylab}"
CREDS_FILE="$HOME/.tinyexplorer-wintest-creds"
# Windows 11 Pro matches the tester's laptop; requires eligible Windows
# licensing (uni M365/E3 qualifies). Fallback that needs no client license:
#   IMAGE="MicrosoftWindowsServer:WindowsServer:2022-datacenter-azure-edition:latest"
IMAGE="${IMAGE:-MicrosoftWindowsDesktop:windows-11:win11-24h2-pro:latest}"

case "${1:-}" in
  up)
    # openssl rather than `tr </dev/urandom | head`: head's early exit would
    # SIGPIPE tr, and set -o pipefail would silently abort the whole script.
    PASSWORD="$(openssl rand -base64 18 | tr -d '/+=')Aa1!"
    umask 177
    printf 'user=%s\npassword=%s\n' "$ADMIN_USER" "$PASSWORD" > "$CREDS_FILE"
    echo "Credentials saved to $CREDS_FILE"

    az group create --name "$RG" --location "$LOC" --output none
    az vm create \
      --resource-group "$RG" --name "$VM" \
      --image "$IMAGE" --size "$SIZE" \
      --admin-username "$ADMIN_USER" --admin-password "$PASSWORD" \
      --public-ip-sku Standard --nsg-rule NONE \
      --output table

    MY_IP="$(curl -fsS https://ifconfig.me)"
    az vm open-port --resource-group "$RG" --name "$VM" \
      --port 3389 --priority 100 \
      --source-address-prefixes "${MY_IP}/32" --output none
    echo "RDP (3389) opened for ${MY_IP}/32 only"

    az vm auto-shutdown --resource-group "$RG" --name "$VM" --time 2200 --output none
    echo "Auto-shutdown set for 22:00 UTC"

    az vm show -d --resource-group "$RG" --name "$VM" \
      --query "{name:name, ip:publicIps, size:hardwareProfile.vmSize}" --output table
    ;;

  test)
    az vm run-command invoke \
      --resource-group "$RG" --name "$VM" \
      --command-id RunPowerShellScript \
      --scripts "@scripts/azure-wintest-remote.ps1" \
      --query "value[].message" --output tsv
    ;;

  rdp)
    IP="$(az vm show -d --resource-group "$RG" --name "$VM" --query publicIps --output tsv)"
    RDP_FILE="/tmp/${VM}.rdp"
    printf 'full address:s:%s:3389\nusername:s:%s\nprompt for credentials:i:1\n' \
      "$IP" "$ADMIN_USER" > "$RDP_FILE"
    echo "IP: $IP — credentials in $CREDS_FILE"
    open "$RDP_FILE"
    ;;

  down)
    az group delete --name "$RG" --yes --no-wait
    echo "Deletion of $RG started (runs in background on Azure)"
    ;;

  *)
    echo "usage: $0 {up|test|rdp|down}" >&2
    exit 1
    ;;
esac
