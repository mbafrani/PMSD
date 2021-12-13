__all__ = ['CreateMDL']


"""
In Vensim, the model file are stored with an extension '.mdl' which is MoDeL format file used to transfer
Vensim models from one platform to another.

The .mdl file format consists of:

•(Optional) Equation Group headers

•Macro definitions

•Regular Equations

•Sketch Information

•Settings


---------------------------------
---First part  we need to care---
---------------------------------

The 'Regular Equations' stores the equation information of a variable entity, and look like:

---- Normal auxiliary variable ---
Characteristic Time=
	10+0.5*Outside Temperature
	~	Minutes
	~		|


variable 'Characteristic Time' is a variable entity in the SDM model, the content in the right part of '=' is the
equation information need to be saved in this variable entity, so that we can run the model

---- Stock variable ----
Teacup Temperature= INTEG (
	-Heat Loss to Room,
		200)
	~
	~		|

With an 'INTEG(a, b)', the variable are denoted as stock variable where 'a' is the equation, i.e., inflow - outflow
and 'b' is the initial value of stock variable

---- Data variable ----
Arrival rate:INTERPOLATE::=
	GET XLS DATA( 'active2012_1D.xlsx','Sheet2','A','B2' )
	~
	~		|

In a SDM, data variables are variables whose value are retrieved directly from local,
so we should call the function : GET XLS DATA(), and therefore, need to follow the syntax of this function as below:

GET XLS DATA('file','tab','time row or col','cell')

Because we know less about the input file, I prefer to leave this as a blank so user need to specify the exact content.

MARK:
                 ~           ~
                 ~ |      or ~
                             |

is the language syntax, need to be fixed somehow like this.
And with my experience, we just need to have these and the context below '~' seems doesn't matter


---------------------------------
---Second part we need to care---
---------------------------------

control information, this part stores some basic information that the model needs in order to get executed
e.g.

********************************************************
	.Control
********************************************************~
		Simulation Control Parameters
	|                                                          //context above this line makes no impact on the model,
	                                                           //just indicates below are control parameters, keep it.

FINAL TIME  = 30        // this should be the final time
	~	Minute
	~	The final time for the simulation.
	|

INITIAL TIME  = 0      // this should be the initial time
	~	Minute
	~	The initial time for the simulation.
	|

SAVEPER  =
        TIME STEP     // don't change this line, simply keep this
	~	Minute [0,?]
	~	The frequency with which output is stored.
	|

TIME STEP  = 0.125  // this should be the time step
	~	Minute [0,?]
	~	The time step for the simulation.
	|


---------------------------------
---Third part we need to care----
---------------------------------

\\\---/// Sketch information - do not modify anything except names    -- keep this line
V300  Do not put anything below this section - it will be ignored     -- keep this line
*View 1                                                               -- view name, and we assume we have one main view, just keep it
$192-192-192,0,Times New Roman|12||0-0-0|0-0-0|0-0-255|-1--1--1|-1--1--1|191,191,100,0  -- defines the views default font and color settings for the view, just keep it.

The remaining lines till a new \\\---/// marker are all interpreted as objects.  Each line begins with 2 numbers.  The first is the object type, the second the object id.

start with 1 <-- this is an arrow, syntax:
1,id,from,to,shape,hid,pol,thick,hasf,dtype,res,color,font,np|plist



10,1,Teacup Temperature,814,555,106,53,3,3,0,0,0,0,0,0
12,2,48,1271,555,10,8,0,3,0,0,-1,0,0,0
1,3,5,2,4,0,0,22,0,0,0,-1--1--1,,1|(1179,555)|  // must have in the .mdl file, and represented as a flow line, shape = 4 means line with right arrow symbol, 100 without left arrow symbol
1,4,5,1,100,0,0,22,0,0,0,-1--1--1,,1|(993,555)| // same as above
11,5,48,1082,555,16,21,34,3,0,0,1,0,0,0         // 11 means valves (flow variable)
10,6,Heat Loss to Room,1082,598,130,21,40,3,0,0,-1,0,0,0
10,7,Room Temperature,1244,738,130,21,8,3,0,0,0,0,0,0
10,8,Characteristic Time,1082,394,130,21,8,3,0,0,0,0,0,0
10,9,Outside Temperature,1717,229,118,33,3,133,0,0,0,0,0,0

normally we will also have some information like below to draw lines between different variables, and shown as curves
in figure, if we skip this, all arrows are shown as direct lines in the graph, so I prefer to skip these and let the use
to do the layout.

1,9,8,5,0,0,0,0,0,64,0,-1--1--1,,1|(408,198)|
1,10,1,6,1,0,0,0,0,64,0,-1--1--1,,1|(340,296)|
1,11,7,6,1,0,0,0,0,64,0,-1--1--1,,1|(437,28


"""

"""
{UTF-8}
Characteristic Time=
	10+0.5*Outside Temperature
	~	Minutes
	~		|

Heat Loss to Room=
	(Teacup Temperature - Room Temperature) / Characteristic Time
	~
	~		|

Room Temperature=
	70+0.5*Outside Temperature
	~
	~		|

Outside Temperature=
	80
	~
	~		|

Teacup Temperature= INTEG (
	-Heat Loss to Room,
		200)
	~	Degrees
	~		|

********************************************************
	.Control
********************************************************~
		Simulation Control Parameters
	|

FINAL TIME  = 30
	~	Minute
	~	The final time for the simulation.
	|

INITIAL TIME  = 0
	~	Minute
	~	The initial time for the simulation.
	|

SAVEPER  =
        TIME STEP
	~	Minute [0,?]
	~	The frequency with which output is stored.
	|

TIME STEP  = 0.125
	~	Minute [0,?]
	~	The time step for the simulation.
	|

\\\---/// Sketch information - do not modify anything except names
V300  Do not put anything below this section - it will be ignored
*View 1
$192-192-192,0,Times New Roman|12||0-0-0|0-0-0|0-0-255|-1--1--1|-1--1--1|191,191,100,0
10,1,Teacup Temperature,814,555,106,53,3,3,0,0,0,0,0,0
12,2,48,1271,555,10,8,0,3,0,0,-1,0,0,0
1,3,5,2,4,0,0,22,0,0,0,-1--1--1,,1|(1179,555)|
1,4,5,1,100,0,0,22,0,0,0,-1--1--1,,1|(993,555)|
11,5,48,1082,555,16,21,34,3,0,0,1,0,0,0
10,6,Heat Loss to Room,1082,598,130,21,40,3,0,0,-1,0,0,0
10,7,Room Temperature,1244,738,130,21,8,3,0,0,0,0,0,0
10,8,Characteristic Time,1082,394,130,21,8,3,0,0,0,0,0,0
10,9,Outside Temperature,1717,229,118,33,3,133,0,0,0,0,0,0


///---\\\
:L<%^E!@
1:Current.vdf
9:Current
22:$,Dollar,Dollars,$s
22:Hour,Hours
22:Month,Months
22:Person,People,Persons
22:Unit,Units
22:Week,Weeks
22:Year,Years
22:Day,Days
15:0,0,0,0,0,0
19:100,0
27:2,
34:0,
4:Time
5:Outside Temperature
35:Date
36:YYYY-MM-DD
37:2000
38:1
39:1
40:6
41:0
42:1
24:0
25:30
26:30


"""
