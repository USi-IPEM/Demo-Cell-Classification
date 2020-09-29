- Data collection organized by files (date) <yymmdd>

- zz-Archiv: folder archiv

- The data is divided by assembly samples:
  - The sample starts when the robots is in the "storage position" about to pick up the white disk
  - The sample ends just before the previous condition is true
  - The sample contains two "quality values". The first one is the left-over of the previous assambly, it must be ignored.
    The second one is the real value of the "quality control"
 
 
- A vector of interest, it is important to consider:
  - Robot position x: see as variable 527
  - Robot position y: see as variable 528
  - Robot position z: see as variable 529
  - Conveyor speed: see as low (300:390) / Fast (390:750) / Too fast (750:950)
  - Quality of the piece: see DistanceAbs

- The piece is considered as OK part as long as the "DistanceAbs" < 1.5
