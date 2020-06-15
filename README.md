# PMSD: Data-Driven Simulation of Business Processes Using Process Mining and System Dynamics 
Forward-looking approach in process mining (using system dynamics and process mining to simulate business processes at an aggregated level).
Process mining field has been enriched with multiple techniques and tools for different purposes. However, these techniques are mostly 'backward-looking'. 
'PMSD' is a web application tool that supports forward-looking simulation techniques. It transforms the event data and process mining results into a simulation model/
## Running the application 
- Run app.py and use the "127.0.0.1:5000" as the homepage URL in the browser. 
- The detail information and the tool tutorial are provided in **PMSDToolTutorial.pdf**. 
- The modules are using the output of the previous step as infrastructure and inputs, therefore each should be used as the provided order in the application for the better result.
## Inside event log
- In this section, event logs in both '.csv' and '.xes' formats can be uploaded.
- The main information of the log is shown and the option to indicate the main attributes are provided.
    - It is recommended to use the event logs with **both** start and complete timestamp. In case of only complete timestamp set both start and complete timestamp to the **existing timestamp** in the event log.
- The generated event log which is stored in the 'Output' folder and can be saved using the link. 
- The process model in the form of a directly-follows graph is discovered and shown. 
##  Time window stability test
- Use different patterns that seem to be suitable for the event log, i.e., the event log shows the most similar behavior. 
    -  Make sure that one aspect (like general) is selected. 
- The patterns are assessed and the errors based on time series analyses are shown.
- Both complete system dynamics log (SD-Log) and active SD-Log (only including active steps after removing periodic inactive time steps such as weekends) are stored in the 'Output' file.
## Event log to sd log
- Using this tab, an sd log can be generated for different aspects. 
- Sd logs can also be used from time window stability step directly for the next steps.
- Statistical analyses of sd log in the given time window and the selected aspects such as closest distribution are calculated and shown. 
## Relation detection
- Using one of the generated sd logs in previous steps, and the set threshold for strength of a relation, the linear and nonlinear relationships between variables are examined.
- The time window shift gives you the option to assess the linear/nonlinear relations between variables in different time steps, e.g., the effect of arrival rate in the first hour on the average waiting time in the third hour. 
## Detailed relation
- For all the detected variables in the previous step, the detailed relations are provided. 
## CLD creation 
- The relations based on their strengths are shown and the user can check which are affected by others. 
- The causal loop diagram (conceptual model) is generated in the application and the 'ModelsFormat' folder. 
    - The generated CLD is in the form of '.mdl' file, which can be uploaded and used in system dynamics software such as Venism. 
## SFD creation
- Based on the number of variables inside the sd log, a specific number of stocks and in/outflows are generated. 
-  The stock-flow diagram (simulation model) is generated in the application and the 'ModelsFormat' folder. 
    - The generated SFD is in the form of '.mdl' file, which can be uploaded and used in system dynamics software such as Venism. 
## Simulation and validation 
- Using the sd log and corresponding SFD model which are generated through the previous steps, the simulation and validation are performed for each variable.
    - The option to see the simulation results are provided by selecting the variable's name. 

