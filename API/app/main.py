from typing import Any, Dict, AnyStr, List, Union
from fastapi import FastAPI, File, UploadFile
from math import *
from json import JSONEncoder
import numpy
import inf
import uvicorn
class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)
app = FastAPI()
JSONObject = Dict[AnyStr, Any]
JSONArray = List[Any]
JSONStructure = Union[JSONArray, JSONObject]
##########################################################

global model

@app.on_event("startup")
def load_clf():
    '''
    load the model in the startup, load only once 
    :return: 
    '''
    global model
    model = inf.initialize_model("./Weights/model_last.ts")
############################## json file #########################################
@app.get("/alive")
def alive():
    '''
    check if the server is alive or no as get request not post request
    :return: {"status":"alive"} otherwise will be page not reachable
    '''
    return {"status":"alive"}

@app.post("/alive")
def alive():
    '''
    check if the server is alive as post request
    :return: {"status":"alive"} otherwise will be page not reachable
    '''
    return {"status":"alive"}

@app.post("/inf/")
async def inference_api(uploaded_file: UploadFile = File(...)):
    '''
    main function for inference api, take a file and as we expect utopia the file i expect 
    will be image.txt contain str of the image
    :param uploaded_file: text file contain the image encoded as str
    :return: 
    {"Detected letters": result,
                "Probability":conf} on success 
    otherwise
    {"Result":"ISSUE With This Image, make sure it's RGB, base64 image",
                "Time":"0"}
    '''
    try:
        file_as_byte = uploaded_file.file.read()
        img=inf.read_base64_image_from_str(file_as_byte)
        result,conf=inf.inference_model(img,model)

        return {"Detected letters": result,
                "Probability":conf}

    except:
        return {"Result":"ISSUE With This Image, make sure it's RGB, base64 image",
                "Time":"0"}

# this is because if you want to run the file directly not as uvicorn command in docker
# if __name__ == "__main__":
#     uvicorn.run("main:app", port=5000, log_level="info")
