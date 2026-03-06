from openreward.environments import Server
from env import SecretMafiaEnvironment

if __name__ == "__main__":
    server = Server([SecretMafiaEnvironment])
    server.run()
