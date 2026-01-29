# PocketBase Setup Guide

## Create Collection
log in to your PocketBase Admin UI and create a new collection named `datasets`:
* **Collection Name:** `datasets`
* **Fields:**
    - `name` (Text, Non-empty)
    - `status` (Select: `pending`, `training`, `ready`, `failed`) -> Default: `pending`
    - `images` (File, Multiple files allowed, Allowed MIME types: image/png, Max file size: 10485760  bytes, Max select: 99)
    - `classifier_file` (File, Single file) -> This is where the Dagster pipeline will upload the ONNX result.

## Create a User
Create a user account in PocketBase Admin UI to be used by the Dagster pipeline to upload files and update records.
