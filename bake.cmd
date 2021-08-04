@echo off

cargo build --bin bake --release
set BAKE=target\release\bake

%BAKE% --scene "assets/meshes/flying_world_-_battle_of_the_trash_god/scene.gltf" --scale 0.001875 -o battle
%BAKE% --scene "assets/meshes/336_lrm/scene.gltf" --scale 0.01 -o 336_lrm
%BAKE% --scene "assets/meshes/pica_pica_-_mini_diorama_01/scene.gltf" --scale 0.1 -o pica
%BAKE% --scene "assets/meshes/floor/scene.gltf" --scale 1.0 -o floor
%BAKE% --scene "assets/meshes/testball/scene.gltf" --scale 1.0 -o testball
%BAKE% --scene "assets/meshes/cornell_box/scene.gltf" --scale 2.0 -o cornell_box
%BAKE% --scene "assets/meshes/gas_stations_fixed/scene.gltf" --scale 0.005 -o gas_stations
%BAKE% --scene "assets/meshes/viziers_observation_deck/scene.gltf" --scale 0.0075 -o viziers
%BAKE% --scene "assets/meshes/dp3_homework_4/scene.gltf" --scale 0.05 -o mini_battle
%BAKE% --scene "assets/meshes/painting_xyz_homework/scene.gltf" --scale 0.0025 -o painting_xyz_homework
%BAKE% --scene "assets/meshes/conference/scene.gltf" --scale 1.0 -o conference
%BAKE% --scene "assets/meshes/roughness-scale/scene.gltf" --scale 0.5 -o roughness-scale
%BAKE% --scene "assets/meshes/emissive/triangle.glb" --scale 0.333 -o emissive-triangle