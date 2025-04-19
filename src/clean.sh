#!/bin/bash

# ========= USER CONFIG =========
MAIN_RG="mlops-test-rg"       # Your main resource group
AKS_CLUSTER_NAME="mlops-aks"  # Optional: your AKS cluster name (if used)
LOCATION="eastus"             # Required for checking MC_* group
DELETE_LOG_WORKSPACES=true   # Set to true to delete log analytics too
# ==============================

echo "🔄 Starting Azure cleanup..."

# 1. Delete the main resource group
echo "🚨 Deleting main resource group: $MAIN_RG"
az group delete --name "$MAIN_RG" --yes --no-wait

# 2. Find and delete the AKS-generated MC_* resource group
MC_RG="MC_${MAIN_RG}_${AKS_CLUSTER_NAME}_${LOCATION}"
if az group exists --name "$MC_RG"; then
  echo "🧼 Deleting AKS managed resource group: $MC_RG"
  az group delete --name "$MC_RG" --yes --no-wait
else
  echo "✅ No AKS managed resource group found."
fi

# 3. Delete unattached managed disks
echo "💽 Deleting unattached managed disks..."
for disk in $(az disk list --query "[?managedBy==null].name" -o tsv); do
  rg=$(az disk show --name "$disk" --query resourceGroup -o tsv)
  echo " - Deleting disk: $disk (RG: $rg)"
  az disk delete --name "$disk" --resource-group "$rg" --yes
done

# 4. Delete unassociated public IPs
echo "🌐 Deleting unassociated public IPs..."
for ip in $(az network public-ip list --query "[?ipAddress==null].name" -o tsv); do
  rg=$(az network public-ip show --name "$ip" --query resourceGroup -o tsv)
  echo " - Deleting public IP: $ip (RG: $rg)"
  az network public-ip delete --name "$ip" --resource-group "$rg"
done

# 5. Delete unattached NICs
echo "🔌 Deleting unattached network interfaces..."
for nic in $(az network nic list --query "[?virtualMachine==null].name" -o tsv); do
  rg=$(az network nic show --name "$nic" --query resourceGroup -o tsv)
  echo " - Deleting NIC: $nic (RG: $rg)"
  az network nic delete --name "$nic" --resource-group "$rg"
done

# 6. Optionally delete unused Log Analytics workspaces
if [ "$DELETE_LOG_WORKSPACES" = true ]; then
  echo "📊 Deleting Log Analytics Workspaces..."
  for ws in $(az monitor log-analytics workspace list --query "[].name" -o tsv); do
    rg=$(az monitor log-analytics workspace show --workspace-name "$ws" --query resourceGroup -o tsv)
    echo " - Deleting workspace: $ws (RG: $rg)"
    az monitor log-analytics workspace delete --workspace-name "$ws" --resource-group "$rg" --yes
  done
fi

echo "✅ Cleanup completed! Monitor Azure Portal to confirm deletion is finalized."
