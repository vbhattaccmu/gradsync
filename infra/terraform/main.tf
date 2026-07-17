# gradsync GPU cluster: N GPU VMs on a private VPC, wired so NCCL can talk
# between every node over their internal IPs.
#
# Cost warning: A100 VMs bill by the second and are NOT cheap (~$3-4/GPU/hr).
# `terraform destroy` the moment a run finishes.

terraform {
  required_version = ">= 1.5"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
  zone    = var.zone
}

# Dedicated VPC so we control the firewall precisely.
resource "google_compute_network" "gradsync" {
  name                    = "gradsync-net"
  auto_create_subnetworks = false
}

resource "google_compute_subnetwork" "gradsync" {
  name          = "gradsync-subnet"
  ip_cidr_range = "10.128.0.0/20"
  region        = var.region
  network       = google_compute_network.gradsync.id
}

# NCCL opens many ephemeral ports between ranks, so allow all traffic *within*
# the subnet. This is safe because nothing external can reach these ports.
resource "google_compute_firewall" "internal" {
  name      = "gradsync-allow-internal"
  network   = google_compute_network.gradsync.id
  direction = "INGRESS"
  allow { protocol = "tcp" }
  allow { protocol = "udp" }
  allow { protocol = "icmp" }
  source_ranges = [google_compute_subnetwork.gradsync.ip_cidr_range]
}

# SSH from anywhere (tighten source_ranges to your IP in real use).
resource "google_compute_firewall" "ssh" {
  name          = "gradsync-allow-ssh"
  network       = google_compute_network.gradsync.id
  direction     = "INGRESS"
  allow {
    protocol = "tcp"
    ports    = ["22"]
  }
  source_ranges = ["0.0.0.0/0"]
  target_tags   = ["gradsync"]
}

resource "google_compute_instance" "node" {
  count        = var.node_count
  name         = "gradsync-node-${count.index}"
  machine_type = var.machine_type
  zone         = var.zone
  tags         = ["gradsync"]

  # GPUs require the instance to terminate (not live-migrate) on host maintenance.
  scheduling {
    on_host_maintenance = "TERMINATE"
    automatic_restart   = true
  }

  guest_accelerator {
    type  = var.gpu_type
    count = var.gpu_count
  }

  boot_disk {
    initialize_params {
      image = var.boot_image
      size  = var.disk_size_gb
      type  = "pd-ssd"
    }
  }

  network_interface {
    subnetwork = google_compute_subnetwork.gradsync.id
    access_config {} # ephemeral external IP for SSH / dataset download
  }

  metadata = {
    ssh-keys               = "${var.ssh_user}:${file(pathexpand(var.ssh_pubkey_path))}"
    install-nvidia-driver  = "True"
    node-rank              = tostring(count.index)
    node-count             = tostring(var.node_count)
    startup-script = templatefile("${path.module}/startup.sh.tpl", {
      node_rank  = count.index
      node_count = var.node_count
    })
  }
}
