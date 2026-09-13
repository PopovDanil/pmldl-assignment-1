import os
import shutil
import subprocess
import sys

MODELS_DIR = os.environ.get("MODELS_DIR", "/opt/airflow/models")
MODEL_FILE = "catboost_model.cbm"
CODE_DIR = os.environ.get("CODE_DIR", "/opt/airflow/code")

API_IMAGE = "bike-api"
APP_IMAGE = "bike-app"
API_CONTAINER = "bike-api-container"
APP_CONTAINER = "bike-app-container"
NETWORK = "bike-network"

API_PORT = 8888
APP_PORT = 8501


def run():
    subprocess.check_call([sys.executable, "-m", "pip", "install", "docker", "-q"])
    import docker

    client = docker.from_env()

    model_src = os.path.join(MODELS_DIR, MODEL_FILE)
    api_context = os.path.join(CODE_DIR, "deployment", "api")
    app_context = os.path.join(CODE_DIR, "deployment", "app")

    model_dest = os.path.join(api_context, MODEL_FILE)
    if os.path.exists(model_src):
        shutil.copy2(model_src, model_dest)
    else:
        print(f"WARNING: Model not found at {model_src}, building without model")
        open(model_dest, "w").close()

    _stop_containers_on_port(client, API_PORT, docker)
    _stop_containers_on_port(client, APP_PORT, docker)
    _stop_and_remove(client, API_CONTAINER, docker)
    _stop_and_remove(client, APP_CONTAINER, docker)
    _remove_network(client, NETWORK, docker)

    network = client.networks.create(NETWORK, driver="bridge")

    print("Building API image...")
    client.images.build(path=api_context, tag=API_IMAGE, rm=True)

    print("Building App image...")
    client.images.build(path=app_context, tag=APP_IMAGE, rm=True)

    print("Starting API container...")
    api_container = client.containers.run(
        API_IMAGE,
        name=API_CONTAINER,
        detach=True,
        ports={f"{API_PORT}/tcp": API_PORT},
        network=NETWORK,
    )
    print(f"API container started: {api_container.short_id}")

    print("Starting App container...")
    app_container = client.containers.run(
        APP_IMAGE,
        name=APP_CONTAINER,
        detach=True,
        ports={f"{APP_PORT}/tcp": APP_PORT},
        network=NETWORK,
        environment={"API_URL": f"http://{API_CONTAINER}:{API_PORT}"},
    )
    print(f"App container started: {app_container.short_id}")

    model_status = "baked into image" if os.path.exists(model_src) else "not found (model missing)"
    print(f"Deployment complete. Model: {model_status}")
    print(f"API: http://localhost:{API_PORT}  |  App: http://localhost:{APP_PORT}")


def _stop_containers_on_port(client, port, docker):
    for container in client.containers.list(all=True):
        ports = container.ports
        for bindings in ports.values():
            if bindings:
                for binding in bindings:
                    if binding.get("HostPort") == str(port):
                        print(f"Stopping container {container.name} on port {port}...")
                        try:
                            container.stop()
                        except Exception:
                            pass
                        try:
                            container.remove()
                        except Exception:
                            pass


def _stop_and_remove(client, name, docker):
    try:
        container = client.containers.get(name)
        try:
            container.stop()
        except Exception:
            pass
        container.remove()
        print(f"Removed container: {name}")
    except docker.errors.NotFound:
        pass


def _remove_network(client, name, docker):
    try:
        network = client.networks.get(name)
        network.remove()
        print(f"Removed network: {name}")
    except docker.errors.NotFound:
        pass


if __name__ == "__main__":
    run()
