
# Hybrid Business Process Simulation
A proof of concept example for updating the DES model of a process based on the system dynamics model of the process.
- Open the Jupyter notebook (**HybridSiminPmSharedMaterialforReview.ipynb**) and use the provided steps. 
- The DES and SD models are taken as input.
- For the ease of use and performing experiments with your desired process, the following supports are provided:
- To generate a CPN model of your process and the corresponding SML file using an event log, you can use the following link, in case of any specific scenario you need to change the process model in the CPN Tools. The link only generates a CPN model based on the given event log, further desired changes should be done by the user.
-- https://cpn-model-process-discovery-1.herokuapp.com/generate-cpn-model/
- In case you want to generate your SD Logs and SD models, this tool supports you https://github.com/mbafrani/PMSD
- Note that you need to make sure that there are common variables names, e.g., service time, arrival rate, ..., are in both models that can be updated.
- To run the current example:
- Run the CPN model (it should be a large period enough)
- Run the main method in the script (it will automatically update the CPN model while running)
- The results (before and after updates based on SD results) are captured in one event log continuously.
