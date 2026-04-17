# please install huggingface_hub before running this code

from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="AaronCIH/PIR_tar",
    local_dir="./PIR_tar",
)
