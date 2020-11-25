##### Machine Learning for industrial process control.

In this repository we explore predictive quality control using
machine learning. The industrial demo setting we consider is
shown in the video below:

###### Demo Video:
![Alt Text](demo.gif)

Regarding the data which can be found in the 01_Data folder our code
makes the assumptions documented below.

Quality control
---------------
- qc: 
   - Only intereseted in absolute value.
   - Last value per file counts.
   - if < 1.5 mm quality is ok.

Robot:
------
- 527_x: Robot position in x.
- 528_y: Robot position in y.
- 529_z: Robot position in z.

- 1,052 z position at disc drop.

- 530_Rx: Robot rotation in x. Not relevant.
- 531_Ry: Robot rotation in y. Not relevant.
- 532_Rz: Robot rotation in z. Not relevant.

Conveyor Belt:
--------------
- Speed:
  - 27: conv1
  - 28: conv2
  - 29: conv3
  
  Higher the number, faster the speed. The range goes from 300 to 950 in values.

Flag:
-----

- 561: Flag_grip
  - 0: Open
  - 1: Close
- 560: Flag_pos (position of robot based in certain conditions)
  - 0: Black disk in store
  - 1: Black disk in gate
  - 2: White disk in store
  - 3: White disk in gate


Vector approach
-----------

Input:
  - Drop arm position for white and black discs.
  - Max raw conveyor "speed" value.

Output:
 - Last qc entry in each file.
 - Quality control float (regression) / bool (classification).

Required data:
 - Iris data set ~150.

Sequence approach
-------------

Input:
  - Complete arm pos sequences.
  - Complete belt speed sequences.
  

Output:
 - Quality control float (regression) / bool (classification).
 
Required data (Problem!):
 - RNNs models in the literature use weeks/months of data.

