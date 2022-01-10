val FILE = "./TWTest8-17-5days.csv";
val SEP = ";";
val d = 300.0;
fun list2string([]) = ""|
list2string(x::l) = x ^ (if l=[] then "" else SEP) ^ list2string(l);

fun create_log_file(l) = 
let
   val file_id = TextIO.openOut(FILE)
   val _ = TextIO.output(file_id, list2string(l)) 
   val _ = TextIO.output(file_id, if l = [] then "" else "\n")
in
   TextIO.closeOut(file_id)
end;


fun write_record(l) = 
let
   val file_id = TextIO.openAppend(FILE)
   val _ = TextIO.output(file_id, list2string(l))
   val _ = TextIO.output(file_id, "\n")
   
in
   TextIO.closeOut(file_id)
end;

val attr = ["Case ID", "Activity","type","StartTime","CompleteTime"];
val minute = 60.0;
val hour = 60.0*minute;
val day = 24.0*hour;
val week = 7.0*day;
fun tuesday_jan_1_2019() = 17532.3333*day;
fun Mtime() = ModelTime.time():time;
fun start_time() = tuesday_jan_1_2019();
fun print_start_time() = Date.fmt "%Y-%m-%d %H:%M:%S" (Date.fromTimeLocal(Time.fromReal(start_time())));
fun t2s(t) = Date.fmt "%Y-%m-%d %H:%M:%S" (Date.fromTimeLocal(Time.fromReal(t+start_time())));


(* ARRIVAL TIME DISTRIBUTIONS *)

(* arrival time intensities vary from 0.0 to 1.0 and are the product of three factors: yearly influences, weekly influences, and daily influences *)

fun at_month_intensity(m:string) =
case m of 
 "Jan" => 1.0
|"Feb" => 1.0
|"Mar" => 1.0
|"Apr" => 1.0
|"May" => 1.0
|"Jun" => 1.0
|"Jul" => 1.0
|"Aug" => 1.0
|"Sep" => 1.0
|"Oct" => 1.0 
|"Nov" => 1.0
|"Dec" => 1.0
| _ => 1.0;

fun at_weekday_intensity(d:string) =
case d of 
 "Mon" => 1.0
|"Tue" => 1.0
|"Wed" => 1.0
|"Thu" => 1.0
|"Fri" => 1.0
|"Sat" => 0.0
|"Sun" => 0.0
| _ => 0.0;

fun at_hour_intensity(h:int) =
case h of 
 0 => 0.0
|1 => 0.0
|2 => 0.0
|3 => 0.0
|4 => 0.0
|5 => 0.0
|6 => 0.0
|7 => 0.0
|8 => 0.5
|9 => 0.75
|10 => 1.0
|11 => 1.0
|12 => 0.75
|13 => 1.0
|14 => 1.0
|15 => 1.5
|16 => 1.0
|17 => 0.5
|18 => 0.0
|19 => 0.0
|20 => 0.0
|21 => 0.0
|22 => 0.0
|23 =>0.0
| _ => 0.0;

(* TIME FUNCTIONS *)

fun t2date(t) = Date.fromTimeLocal(Time.fromReal(t+start_time()));


fun t2year(t) = Date.year(t2date(t)):int;
fun t2month(t) = Date.month(t2date(t)):Date.month;
fun t2day(t) = Date.day(t2date(t)):int;
fun t2hour(t) = Date.hour(t2date(t)):int;
fun t2minute(t) = Date.minute(t2date(t)):int;
fun t2second(t) = Date.second(t2date(t)):int;
fun t2weekday(t) = Date.weekDay(t2date(t)):Date.weekday;

fun t2monthstr(t) = Date.fmt "%b" (Date.fromTimeLocal(Time.fromReal(t+start_time())));
fun t2weekdaystr(t) = Date.fmt "%a" (Date.fromTimeLocal(Time.fromReal(t+start_time())));

(*change system time and calculate the difference from one hour*)
fun remaining_time_hour(t) = hour - ((fromInt(t2minute(t))*minute) + fromInt(t2second(t)));



(* overall intensity *)
fun at_intensity(t) = at_month_intensity(t2monthstr(t))*at_weekday_intensity(t2weekdaystr(t))*
at_hour_intensity(t2hour(t));

(* Use this function to sample interarrival times: t is the current time and d is the net delay: It moves forward based on intensities: the lower the intensity, the longer the delay in absolute time.*) 
fun rel_at_delay(t,d) = 
if d < 0.0001
   then 0.0
   else if d < remaining_time_hour(t)*at_intensity(t)
        then d/at_intensity(t)
        else rel_at_delay(t+remaining_time_hour(t),
            d-(remaining_time_hour(t)*at_intensity(t)))+hour; 

(* same but now without indicating current time explicitly *)
fun r_at_delay(d) = rel_at_delay(Mtime(),d);

(* the average ratio between effective/net time (parameter d) and delay in actual time*)
val eff_at_factor = r_at_delay(52.0*week)/(52.0*week);

(* normalized interarrival time delay using the ratio above *)
fun norm_rel_at_delay(t,d) = rel_at_delay(t,d/eff_at_factor) ;


(* normalized  interarrival time delay using the ratio above *)
fun norm_r_at_delay(d) = r_at_delay(d/eff_at_factor) ;

val man = norm_r_at_delay(d);

val Servicetimepercase=1200.0;


fun t(rt:ReqType) = 
case rt:ReqType of 
 car => "NewRequest"
|driver=> "CurrentCustRequest";
 
fun event(i:int,tes:string,ty:CaseReq,re:real,res1:ResServic)= write_record([Int.toString(i),tes, t(#2(ty)),t2s(Mtime()),t2s(Mtime()+re),(#2(res1))]);

fun checktime ()= 
if OS.FileSys.compare(OS.FileSys.fileId("TWTest.sml"),OS.FileSys.fileId("Changed_cpn_model.sml")) = EQUAL
then use "TWTest.sml"
else use "Changed_cpn_model.sml"
;