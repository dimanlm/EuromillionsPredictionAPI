from fastapi import FastAPI


app = FastAPI()

@app.get("/")
async def read_items():
    return 0