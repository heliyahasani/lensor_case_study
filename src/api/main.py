import uvicorn
from src.configs.app import settings

def start():
    uvicorn.run("src.api.api:app", host=settings.app_host, port=settings.app_port, reload=True)

if __name__ == "__main__":
    start()
