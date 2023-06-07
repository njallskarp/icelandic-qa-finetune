import os 

def get_config():
    return {
        "name": "ruquad_raw",

    }

def get_data():

    config = get_config()
    
    os.system(f"curl --output ./datafiles/{['name']}_1.zip https://repository.clarin.is/repository/xmlui/bitstream/handle/20.500.12537/311")
    
    return ([], [], []),  ([], [], [])
    
