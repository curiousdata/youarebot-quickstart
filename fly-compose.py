import json
import yaml
import subprocess
import os
import logging
from dataclasses import dataclass
from typing import List, Dict, Any
import concurrent.futures

logger = logging.getLogger("fly-compose")

# Configure the logging
logging.basicConfig(
    level=logging.DEBUG,  # Set the lowest level of messages to capture
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),  # Log messages to a file
        logging.StreamHandler()         # Log messages to the console
    ]
)
def read_compose():
    files_possible_name = ["compose.yml", "docker-compose.yml"]
    for file_name in files_possible_name:
        if os.path.exists(file_name):
            with open(file_name, "r") as f:
                return f.read()
    print("No compose file found")



@dataclass
class ImageRef:
    registry: str
    repository: str
    tag: str
    digest: str

@dataclass
class GuestConfig:
    cpu_kind: str
    cpus: int
    memory_mb: int

@dataclass
class RestartPolicy:
    policy: str
    max_retries: int

@dataclass
class Config:
    init: Dict[str, Any]
    guest: GuestConfig
    image: str
    restart: RestartPolicy
    dns: Dict[str, Any]

@dataclass
class Event:
    type: str
    status: str
    source: str
    timestamp: int

@dataclass
class Instance:
    id: str
    name: str
    state: str
    region: str
    image_ref: ImageRef
    instance_id: str
    private_ip: str
    created_at: str
    updated_at: str
    config: Config
    events: List[Event]
    host_status: str

@dataclass
class ServiceSpec:
    count: int
    cpu: int
    memory: int
    kind: str
@dataclass
class ServiceVolume:
    name: str
    source: str
    is_created: bool = False
@dataclass
class ServiceEnvirontment:
    key: str
    value: str
@dataclass
class ServicePort:
    host: int
    container: int

def ask_and_execute(command: str):
    logger.info(f"Executing command: {command}")
    exec = subprocess.run(command, text=True, capture_output=True)
    logger.info(f"Command executed: {exec.stdout}")
    return exec
    
class Service:
    instances: List[Instance] = []
    def __init__(self, organization: str,region: str, name: str, image: str, command: str, build: str , envs: List[ServiceEnvirontment], volumes: List[ServiceVolume], ports: List[ServicePort], spec: ServiceSpec):
        self.name = name
        self.organization = organization
        self.region = region
        self.image = image
        
        if self.image == None or self.image == "":
            ### create default machine
            self.image = "alpine"
        self.build = build
        self.command = command
        self.volumes = volumes
        self.envs = envs
        if len(self.volumes) > 0:
            for volume in self.volumes:
                ### check if dot delete
                if volume.name == ".":
                    self.volumes.remove(volume)
        self.spec = spec
        if spec == None:
            self.spec = ServiceSpec(1, 1, 1024, "shared")
        self.is_created = False
        self.ports = ports
        self.check()
        self.check_machine()
    def check(self):
        response = ask_and_execute("fly app list").stdout.split("\n")
        for line in response:
            data = line.split()
            if len(data) > 0 and data[0] == self.name:
                self.is_created = True
        logger.info(f"Service {self.name} is created: {self.is_created}")
    
    def create(self):
        if not self.is_created:
            response = ask_and_execute(f"fly app create {self.name} -o {self.organization}").stdout
    def up(self):
        self.check()
        if not self.is_created:
            self.create()
            
        self.check_machine()
        self.rescale_machine()
        self.rescale_volume()
        self.deploy_machine()
        
    def machine_args(self, port=True, region=True):
        args = []
        args.append(f"--app {self.name}")
        args.append(f"--vm-cpu-kind {self.spec.kind}")
        args.append(f"--vm-cpus {self.spec.cpu}")
        args.append(f"--vm-memory {self.spec.memory}")
        if region:
            args.append(f"--region {self.region}")
        for env in self.envs:
            args.append(f"--env {env.key}={env.value}")
        if port:   
            for ports in self.ports:
                args.append(f"--port {ports.container}/tcp")
        for volume in self.volumes:
            args.append(f"--volume {volume.name}:{volume.source}")
        return " ".join(args)
            
    def deploy_machine(self):
        if self.build is not None:
            response = ask_and_execute(f"fly deploy {self.build} {self.machine_args(port=False,region=False)}").stdout
            print(response)
        else:
            for deployment in range(self.spec.count):
                response = ask_and_execute(f"fly machine run {self.image} {self.machine_args(region=True)}").stdout
                print(response)
    def check_volume(self):
        if not self.is_created:
            return
        response = ask_and_execute(f"fly volume list --app {self.name} --json").stdout
        json_parsed = json.loads(response)
        for volume_data in json_parsed:
            name = volume_data.get('name')
            for volume in self.volumes:
                if volume.name == name:
                    volume.is_created = True
        logger.info(f"{len(self.volumes)} volumes checked")
    def rescale_volume(self):
        self.check_volume()
        for volume in self.volumes:
            if not volume.is_created:
                response = ask_and_execute(f"fly volume create {volume.name} --app {self.name} --count {self.spec.count} -y --region {self.region} --json").stdout
                logger.info(f"Volume {volume.name} created")
    def rescale_machine(self):
        ## TODO : should destroy if changes in spec count, volumes, and ports
        self.check_machine()
        logger.info(f"Destroying Previous Machines")
        for instance in self.instances:
            print("TEST")
            print(instance)
            try:
                 response = ask_and_execute(f"fly machine destroy {instance['id']} --app {self.name} --force")
            except Exception as e:
                logger.info(f"Failed to destroy" + str(e))
                
            logger.info(f"Machine {instance['id']} destroyed")

        
    
    def check_machine(self):
        if not self.is_created:
            logger.info(f"Cannot Check Machine Because Service {self.name} not created yet")
            return
        self.instances = json.loads(subprocess.run(f"fly machine list --app {self.name} --json", capture_output=True, text=True).stdout)
        
    def to_json(self):
        return {
            "build": self.build,
            "image": self.image,
            "volumes": self.volumes,
            "command": self.command,
            "ports": self.ports,
            "environtments": self.envs,
            "resources": {
                "kind": self.spec.kind,
                "count": self.spec.count,
                "limits": {
                    "cpu": self.spec.cpu,
                    "memory": self.spec.memory
                }
            }
        }

class Infra:
    services = []
    preffix = ""
    organization = ""
    region = ""
    
    def __init__(self, yamlString: str):
        self.data = yaml.safe_load(yamlString)
        self.organization = self.data.get('fly_organization')
        self.region = self.data.get('fly_region')
        if self.region == None:
            self.region = "lhr"
        if self.organization == None:
            self.organization = "personal"
        self.preffix = self.data.get('fly_preffix_app')
        logger.info(f"Docker Compose file loaded")
        logger.info(f"Preffix: {self.preffix}")
        for service in self.data.get('services'):
            self.register_service(service)
    def fly_check_cli_check(self):
        response = ask_and_execute("fly version").returncode
        if response != 0:
            print("Fly cli not found")
            exit(1)
    def register_service(self, service_name: str):
        serviceConfig = self.data.get('services').get(service_name)
        environments: List[ServiceEnvirontment] = []
        volumes: List[ServiceVolume] = []
        if serviceConfig.get('environment'):
            for env in serviceConfig.get('environment'):
                environments.append(ServiceEnvirontment(env, serviceConfig.get('environment').get(env)))
        if serviceConfig.get('volumes'):
            for volume in serviceConfig.get('volumes'):
                data = volume.split(":")
                volumes.append(ServiceVolume(data[0], data[1]))
        spec: ServiceSpec = ServiceSpec(1, 1, 1024, "shared")
        if serviceConfig.get('resources'):
            if serviceConfig.get('resources').get('kind'):
                spec.kind = serviceConfig.get('resources').get('kind')
            if serviceConfig.get('resources').get('count'):
                spec.count = serviceConfig.get('resources').get('count')
            if serviceConfig.get('resources').get('limits'):
                spec.cpu = serviceConfig.get('resources').get('limits').get('cpus')
                spec.memory = serviceConfig.get('resources').get('limits').get('memory') 
        ports: List[ServicePort] = []
        if serviceConfig.get('ports'):
            for port in serviceConfig.get('ports'):
                data = port.split(":")
                ports.append(ServicePort(data[0], data[1]))
        self.services.append(Service(
                    self.organization,
                    self.region,
                    self.preffix + "-" + service_name, 
                    serviceConfig.get('image'), 
                                        serviceConfig.get('command'),
                    serviceConfig.get('build'), 
                    environments,
                    volumes,
                    ports,
                    spec
                    ))
        
        print(f"Service {service_name} registered")
        
                
        

infra = Infra(read_compose())


with concurrent.futures.ThreadPoolExecutor() as executor:
    futures = [executor.submit(service.up) for service in infra.services]
    for future in concurrent.futures.as_completed(futures):
        print("----")
print("All services deployed")
exit(0)
