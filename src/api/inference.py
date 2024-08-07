from fastapi import APIRouter
import os
import aiofiles
from fastapi import UploadFile, File,Request

class BaseService:
    def __init__(self):
        pass
class CreateInferenceService(BaseService):
    async def save_image_file(self, file: UploadFile = File(...)):
        # Save the file directly in the current working directory
        # Use file.filename to get the filename
        file_path = os.path.join(os.getcwd(), file.filename.strip())

        # Save the file asynchronously
        async with aiofiles.open(file_path, "wb") as out_file:
            content = await file.read()  # Read the file content
            await out_file.write(content)  # Write to the file
        return file_path

    async def execute(self, file: UploadFile) -> dict:
        file_path = await self.save_image_file(file)
        try:
            pass
        finally:
            # Cleanup: Remove the file after processing
            os.remove(file_path)
        return "inference"

router = APIRouter()


@router.get("/inference")
async def inference(request: Request, file: UploadFile)):
    service = CreateInferenceService()
    inference = await service.execute(file)
    return inference
