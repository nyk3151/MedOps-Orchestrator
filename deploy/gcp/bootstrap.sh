#!/bin/bash
# One-time GCP setup, run by a project Owner — ideally in Cloud Shell (already authenticated,
# so nothing needs `gcloud auth` on the workstation):
#
#   PROJECT_ID=<project> bash deploy/gcp/bootstrap.sh      (REGION defaults to asia-southeast1)
#
# It enables the APIs and creates four service accounts with least-privilege roles:
#   medops-deployer : used by deploy/gcp/deploy.py (key file -> keep outside the repo)
#   medops-ui       : runtime of the UI/agent service   (Vertex AI user, Cloud SQL client),
#                     served behind IAP (only allowed Google accounts can open it)
#   medops-api      : runtime of the job server service (Cloud SQL client, runs GPU jobs)
#   medops-gpu      : runtime of the GPU analysis job   (bucket access only, granted at deploy)
# Bucket-level permissions are granted by deploy.py when it creates the bucket.
set -euo pipefail

: "${PROJECT_ID:?set PROJECT_ID}"
REGION="${REGION:-asia-southeast1}"  # Cloud Run GPU (L4) is not offered in asia-northeast1
GPU_REGION="$REGION"
KEY_OUT="${KEY_OUT:-$HOME/medops-deployer-sa.json}"

gcloud config set project "$PROJECT_ID" >/dev/null

echo "== enabling APIs"
gcloud services enable \
  run.googleapis.com sqladmin.googleapis.com artifactregistry.googleapis.com \
  storage.googleapis.com secretmanager.googleapis.com iam.googleapis.com \
  cloudresourcemanager.googleapis.com aiplatform.googleapis.com serviceusage.googleapis.com \
  iap.googleapis.com

sa() { echo "$1@${PROJECT_ID}.iam.gserviceaccount.com"; }
create_sa() {
  gcloud iam service-accounts describe "$(sa "$1")" >/dev/null 2>&1 \
    || gcloud iam service-accounts create "$1" --display-name "$2"
}
bind() {  # bind <member> <role>
  gcloud projects add-iam-policy-binding "$PROJECT_ID" --member "$1" --role "$2" \
    --condition=None >/dev/null
  echo "   $2 -> $1"
}

echo "== service accounts"
create_sa medops-deployer "Med-Ops deployer"
create_sa medops-ui "Med-Ops UI / agent runtime"
create_sa medops-api "Med-Ops job server runtime"
create_sa medops-gpu "Med-Ops GPU analysis job runtime"

echo "== roles"
D="serviceAccount:$(sa medops-deployer)"
for role in roles/run.admin roles/cloudsql.admin roles/artifactregistry.admin \
            roles/storage.admin roles/secretmanager.admin roles/serviceusage.serviceUsageConsumer \
            roles/iap.admin roles/iap.settingsAdmin roles/oauthconfig.editor \
            roles/logging.viewer; do   # logging.viewer: read Cloud Run logs when a deploy fails
  bind "$D" "$role"
done
# The deployer may attach (only) the three runtime accounts to the services/jobs it deploys.
for rt in medops-ui medops-api medops-gpu; do
  gcloud iam service-accounts add-iam-policy-binding "$(sa $rt)" \
    --member "$D" --role roles/iam.serviceAccountUser >/dev/null
  echo "   serviceAccountUser on $rt -> deployer"
done

bind "serviceAccount:$(sa medops-ui)"  roles/aiplatform.user
bind "serviceAccount:$(sa medops-ui)"  roles/cloudsql.client
bind "serviceAccount:$(sa medops-api)" roles/cloudsql.client
bind "serviceAccount:$(sa medops-api)" roles/run.developer      # execute the GPU job with overrides
gcloud iam service-accounts add-iam-policy-binding "$(sa medops-gpu)" \
  --member "serviceAccount:$(sa medops-api)" --role roles/iam.serviceAccountUser >/dev/null
echo "   serviceAccountUser on medops-gpu -> medops-api"
# signed download links for Slicer scenes (~200 MB, over Cloud Run's 32 MiB response cap):
# the job server signs them as itself through the IAM signBlob API (no key file)
gcloud iam service-accounts add-iam-policy-binding "$(sa medops-api)" \
  --member "serviceAccount:$(sa medops-api)" --role roles/iam.serviceAccountTokenCreator >/dev/null
echo "   serviceAccountTokenCreator on medops-api -> itself"

echo "== deployer key -> $KEY_OUT  (download it; never commit it)"
gcloud iam service-accounts keys create "$KEY_OUT" --iam-account "$(sa medops-deployer)"

echo
echo "IAP (UI access restricted to allowed Google accounts): in a project without an organization"
echo "the first enablement needs an OAuth consent screen (User type: External) and may have to be"
echo "done once in the Console: https://console.cloud.google.com/security/iap?project=${PROJECT_ID}"
echo
echo "Done. Cloud Run GPU (nvidia-l4) quota in ${GPU_REGION}:"
echo "  https://console.cloud.google.com/iam-admin/quotas?project=${PROJECT_ID} (filter: 'Cloud Run' + 'L4')"
echo "Place the key at C:/Users/<you>/.gcp/medops-deployer-sa.json and set MEDOPS_DEPLOY_CREDENTIALS to it."
