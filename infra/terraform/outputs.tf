output "node_names" {
  value = google_compute_instance.node[*].name
}

output "external_ips" {
  description = "SSH into these."
  value       = google_compute_instance.node[*].network_interface[0].access_config[0].nat_ip
}

output "internal_ips" {
  description = "Use node 0's internal IP as --master-addr when launching."
  value       = google_compute_instance.node[*].network_interface[0].network_ip
}

output "master_addr" {
  description = "Pass this to `python -m gradsync.launch --master-addr ...`."
  value       = google_compute_instance.node[0].network_interface[0].network_ip
}

output "launch_hint" {
  value = <<-EOT
    # On node 0:
    python -m gradsync.launch --nnodes ${var.node_count} --node-rank 0 \
      --nproc-per-node ${var.gpu_count} \
      --master-addr ${google_compute_instance.node[0].network_interface[0].network_ip} \
      --master-port 29500 examples/train_mnist.py
    # On each other node N: same command with --node-rank N
  EOT
}
