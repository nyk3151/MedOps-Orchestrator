# Cloud PACS: Orthanc + Orthanc Explorer 2 + Stone Web Viewer, viewer accounts read-only.
#   docker build -f deploy/pacs/Dockerfile -t medops-pacs:dev .
# Configuration (DB, accounts) comes from environment variables set by deploy/gcp/deploy.py.
FROM orthancteam/orthanc:26.9.0
COPY deploy/pacs/readonly.lua /etc/orthanc/medops-readonly.lua
ENV ORTHANC__LUA_SCRIPTS='["/etc/orthanc/medops-readonly.lua"]' \
    DICOM_WEB_PLUGIN_ENABLED=true \
    STONE_WEB_VIEWER_PLUGIN_ENABLED=true \
    ORTHANC__STONE_WEB_VIEWER__SHOW_INFO_PANEL_AT_STARTUP=Never \
    ORTHANC__ORTHANC_EXPLORER_2__UI_OPTIONS__ENABLE_UPLOAD=false \
    ORTHANC__ORTHANC_EXPLORER_2__UI_OPTIONS__ENABLE_DELETE_RESOURCES=false \
    ORTHANC__ORTHANC_EXPLORER_2__UI_OPTIONS__ENABLE_MODIFICATION=false \
    ORTHANC__ORTHANC_EXPLORER_2__UI_OPTIONS__ENABLE_ANONYMIZATION=false \
    ORTHANC__ORTHANC_EXPLORER_2__UI_OPTIONS__ENABLE_SEND_TO=false \
    ORTHANC__ORTHANC_EXPLORER_2__UI_OPTIONS__ENABLE_SETTINGS=false \
    ORTHANC__ORTHANC_EXPLORER_2__UI_OPTIONS__ENABLE_ADD_SERIES=false
