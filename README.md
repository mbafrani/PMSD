# PMSD: Data-Driven Simulation of Business Processes Using Process Mining and System Dynamics 
Forward-looking approach in process mining (using system dynamics and process mining to simulate business processes at an aggregated level).
Process mining field has been enriched with multiple techniques and tools for different purposes. However, these techniques are mostly 'backward-looking'. 
'PMSD' is a web application tool that supports forward-looking simulation techniques. It transforms the event data and process mining results into a simulation model/
## Running the application 
- Run app.py and use the "127.0.0.1:5000" as the homepage URL in the browser.
## Inside event log
- In this section, event logs in both '.csv' and '.xes' formats can be uploaded.
- The main information of the log is shown and the option to indicate the main attributes are provided.
-- It is recommended to use the event logs with both start and complete timestamp. In case of only compelte timestamp set both start and complete timestamp to the existing timestamp in the event log.
- The genrated event log which is stored in 'Output' folder and can be saved using the link. 
- The process model in the form of directly-follows graph is discoverd and shown. 
##  Time window stability test
- Use different patterns that seems to be suitable for the event log, i.e., the event log shows the most similar behavior. 
- The pattern are assessed and the errors based on time series analyses are shown.
- Both compelte system dynamics log (SD-Log) and active SD-Log (only including active steps after removing periodic inactive time steps such as weekends) are stored in 'Output' file.
## Event log to sd log
## Relation detection
## Detailed relation 
## CLD creation 
## SFD creation
## Simulation and validation 