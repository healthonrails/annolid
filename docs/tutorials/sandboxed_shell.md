# Using the Sandboxed Shell Tool in Annolid

Annolid now features a **Sandboxed Exec Tool** that intelligently runs shell commands requested by the AI inside of an isolated Docker container rather than directly on your host machine. This protects your system from accidental command mistakes or complex dependencies while allowing the AI to run powerful tools.

## How It Works

By default, when Annolid attempts to run a shell command on your behalf, it checks your system for `docker`.

- **If Docker is running**: Annolid automatically pulls the official `ubuntu:24.04` image and runs the command inside it. It mounts your **current workspace directory** into the container so the AI can still read code and output files, but the rest of your computer is completely safe.
- **If Docker is missing**: Annolid operates normally, falling back to safe local host command execution using regex guards (like blocking `rm -rf`).

You do not need to do any manual configuration within Annolid for this to work!

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

Once Docker is running, open the Annolid Bot chat and give it a CLI-style task. For example:

> *"Run a command to create a new folder named 'test_container', then print 'Hello from Docker' inside it."*

If the container isolation is working, you will see a flash of the `docker run` command in the agent's internal reasoning logs. If you open your system's terminal, you can temporarily run `docker ps` while Annolid is "thinking" to actually see the transient Ubuntu container alive and working!

## Troubleshooting

- **"falling back to local executor" error:** If you see logging indicating the sandbox was bypassed, ensure Docker Desktop is actively running and that the `docker` command is available in your PATH.
- **Missing dependencies in the container:** The container launches as a relatively empty Ubuntu environment. If the AI needs `python3` or `git` inside the container, it is smart enough to run `apt-get install` internally to quickly get what it needs. However, the exact state disappears after each command runs (they are stateless commands).
- **Files disappear:** Remember that the Sandbox *only mounts your current Annolid workspace*. If the AI creates temporary files inside `/tmp` of the container, you will not see them on your host. If it needs to save something for you, tell it specifically to save the file inside your project directory.
