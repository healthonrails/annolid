# Using the Sandboxed Shell Tool in Annolid

Annolid provides a **Sandboxed Exec Tool** that runs shell commands requested by the AI inside an isolated Docker container rather than directly on your host machine.

## How It Works

By default, Annolid restricts runtime tools to the configured workspace and
checks your system for `docker` before executing a shell command.

- **If Docker is running**: Annolid runs the command with a reviewed official Ubuntu 24.04 image pinned to an immutable multi-architecture SHA-256 digest. It mounts your **current workspace directory** read-only and applies network, capability, process, privilege, and temporary-filesystem restrictions.
- **If Docker is missing or unavailable**: the sandboxed command fails closed. Annolid does not silently run it on the host.

The workspace mount is read-only by default. Commands can use the container's
temporary `/tmp`, but they cannot modify workspace files through this tool.

The secure defaults are equivalent to:

```json
{
  "tools": {
    "restrictToWorkspace": true,
    "exec": {
      "containerImage": "ubuntu:24.04@sha256:4fbb8e6a8395de5a7550b33509421a2bafbc0aab6c06ba2cef9ebffbc7092d90"
    }
  }
}
```

The image reference must end in `@sha256:` followed by a 64-character digest;
floating tags fail closed. Updating the sandbox base is therefore an explicit,
reviewable config or source change.

Workspace restriction removes the host-backed `exec_start` and `exec_process`
tools from the registered tool set. An existing config can explicitly set
`restrictToWorkspace` to `false` for compatibility, but the security audit
reports that configuration as high risk.

## Setup Instructions

To enable container isolation, you must install and start Docker on your computer:

### macOS

1. Download **Docker Desktop for Mac** from [docker.com](https://www.docker.com/products/docker-desktop/).
2. Run the installer and move Docker to your Applications folder.
3. Open Docker Desktop and follow the setup wizard.
4. Keep the Docker app running in your menu bar when using Annolid.

### Windows (WSL 2)

1. Install **Docker Desktop for Windows**.
2. Ensure you have the WSL 2 backend enabled during installation.
3. Keep Docker Desktop running when you use Annolid.

### Linux (Ubuntu/Debian)

1. Run the following commands in your terminal:

   ```bash
   sudo apt-get update
   sudo apt-get install docker.io
   sudo systemctl enable --now docker
   sudo usermod -aG docker $USER
   ```

2. You will need to log out and log back in for the user group changes to take effect.

## Verifying It Works

Once Docker is running, open the Annolid Bot chat and give it a read-only CLI-style task. For example:

> *"Run a command to list the Python files in this workspace."*

If the container isolation is working, you will see a flash of the `docker run` command in the agent's internal reasoning logs. If you open your system's terminal, you can temporarily run `docker ps` while Annolid is "thinking" to actually see the transient Ubuntu container alive and working!

## Troubleshooting

- **"Sandbox unavailable" error:** Ensure Docker Desktop or the Docker daemon is running and that the `docker` command is available in your `PATH`.
- **Missing dependencies in the container:** The pinned Ubuntu image is intentionally minimal, runs without network access, and is stateless. It cannot install packages during a command. Use Annolid's typed tools or configure a reviewed custom image pinned by digest.
- **"Sandbox image must be pinned" error:** replace a floating image tag with a reviewed digest reference. Run `annolid-run agent-security-audit` to find unsafe runtime configuration.
- **Files disappear:** The sandbox's `/tmp` is temporary and the workspace mount is read-only. Use Annolid's workspace file tools for reviewed file changes.
