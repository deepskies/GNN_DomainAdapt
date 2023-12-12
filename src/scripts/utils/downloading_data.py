import requests
from .paths import data, datapath

# Modify these parameters to decide what simulations to download
sims = ["IllustrisTNG", "SIMBA"]
indexes = range(1000)

"""Download the dataset from Flatiron Institute.
e.g. link: https://users.flatironinstitute.org/~camels/FOF_Subfind/IllustrisTNG/LH/LH_567/fof_subhalo_tab_005.hdf5 

Args:
    sims (lst): strings of simulations, "IllustrisTNG" or "SIMBA"
    indexes (ndarray): Indexes of subfind data from LH dataset to be downloaded.
"""

# Create the data directory if it doesn't exist
data.mkdir(parents=True, exist_ok=True)

destination = datapath
url_prefix = "https://users.flatironinstitute.org/~camels/FOF_Subfind/"
suffix = "fof_subhalo_tab_033.hdf5"

seeds_Illutris = "https://users.flatironinstitute.org/~camels/Sims/IllustrisTNG/CosmoAstroSeed_params.txt"
seeds_SIMBA = "https://users.flatironinstitute.org/~camels/Sims/SIMBA/CosmoAstroSeed_params.txt"

with open(destination + "CosmoAstroSeed_params_IllustrisTNG.txt", "wb") as f:
    f.write(requests.get(seeds_Illutris).content)

with open(destination + "CosmoAstroSeed_params_SIMBA.txt", "wb") as f:
    f.write(requests.get(seeds_SIMBA).content)

for sim in sims:
    for i in indexes:
        url = url_prefix + sim + "/LH/LH_" + str(i) + "/" + suffix
        name = destination + sim + "_LH_" + str(i) + "_" + suffix
        r = requests.get(url)
        f = open(name, 'wb')
        f.write(r.content)
        print(f"File downloaded for {sim} set {i}")
        f.close()
