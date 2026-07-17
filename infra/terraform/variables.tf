variable "project_id" {
  type        = string
  description = "GCP project id. Must have Compute Engine API enabled and GPU quota."
}

variable "region" {
  type    = string
  default = "us-central1"
}

variable "zone" {
  type        = string
  default     = "us-central1-a"
  description = "Zone must have the requested GPU. Check with `gcloud compute accelerator-types list`."
}

variable "node_count" {
  type        = number
  default     = 2
  description = "Number of GPU VMs. Start with 2 to demo cross-node NCCL."
}

variable "machine_type" {
  type    = string
  # a2-highgpu-1g = 1x A100 40GB. For multi-GPU-per-node (to exercise NVLink
  # inside a node) use a2-highgpu-2g / a2-ultragpu-2g and raise gpu_count.
  default = "a2-highgpu-1g"
}

variable "gpu_type" {
  type    = string
  default = "nvidia-tesla-a100"
}

variable "gpu_count" {
  type    = number
  default = 1
}

variable "boot_image" {
  type = string
  # Deep Learning VM: ships NVIDIA driver + CUDA + NCCL + PyTorch, so the demo is
  # about gradsync, not driver-wrangling. The startup script just adds gradsync.
  default = "projects/deeplearning-platform-release/global/images/family/common-cu121-debian-11"
}

variable "disk_size_gb" {
  type    = number
  default = 100
}

variable "ssh_user" {
  type        = string
  description = "Username to inject the SSH public key for."
}

variable "ssh_pubkey_path" {
  type    = string
  default = "~/.ssh/id_ed25519.pub"
}
