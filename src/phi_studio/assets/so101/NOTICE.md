# SO-101 model

`model.json` and `meshes.bin` are compiled by `scripts/build_so101_model.py` from the SO-101 MJCF
(`so101_new_calib.xml`) and its STL meshes in TheRobotStudio's SO-ARM100 repository,
<https://github.com/TheRobotStudio/SO-ARM100> (`Simulation/SO101`), generated from the SO-ARM100 Onshape CAD by
onshape-to-robot. Licensed under the Apache License 2.0.

Changes: visual meshes only (collision meshes dropped), vertices deduplicated and quantised to uint16 inside each
mesh's bounding box (error under 1.1 micrometres), normals precomputed and split along edges sharper than 36 degrees.
The kinematic tree, joint axes, joint ranges and the `gripperframe` tool site are unchanged.
