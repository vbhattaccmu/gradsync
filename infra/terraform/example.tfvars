# Copy to terraform.tfvars and fill in. Then: terraform init && terraform apply
project_id      = "your-gcp-project"
zone            = "us-central1-a"
node_count      = 2
machine_type    = "a2-highgpu-1g"
gpu_type        = "nvidia-tesla-a100"
gpu_count       = 1
ssh_user        = "your-username"
ssh_pubkey_path = "~/.ssh/id_ed25519.pub"
