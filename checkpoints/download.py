# please install huggingface_hub before running this code

from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="AaronCIH/RAR_modelzoo",
    local_dir="./RAR_modelzoo",
)
