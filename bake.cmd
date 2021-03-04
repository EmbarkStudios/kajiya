@echo off

set SMOL_THREADS=64

rem cargo run --bin bake --release -- --scene "assets/meshes2/sci-fi_floor_panel/scene.gltf" --scale 2.0 -o sci-fi_floor_panel
rem cargo run --bin bake --release -- --scene "assets/meshes3/draft_punk/scene.gltf" --scale 0.1 -o sheds
rem cargo run --bin bake --release -- --scene "assets/meshes3/flying_world_-_battle_of_the_trash_god/scene.gltf" --scale 0.0025 -o battle
rem cargo run --bin bake --release -- --scene "assets/meshes2/rm_342/scene.gltf" --scale 0.015 -o rm_342
rem cargo run --bin bake --release -- --scene "assets/meshes2/336_lrm/scene.gltf" --scale 0.015 -o 336_lrm
rem cargo run --bin bake --release -- --scene "assets/meshes/sploosh-o-matic/scene.gltf" --scale 0.1 -o sploosh-o-matic
rem cargo run --bin bake --release -- --scene "assets/meshes3/sponza/Sponza.gltf" --scale 0.32 -o sponza
rem cargo run --bin bake --release -- --scene "assets/meshes/pica_pica_-_mini_diorama_01/scene.gltf" --scale 0.1 -o pica
rem cargo run --bin bake --release -- --scene "assets/meshes2/b0x-bot/scene.gltf" --scale 1.0 -o b0x-bot

cargo run --bin bake --release -- --scene "assets/test-meshes/floor.gltf" --scale 1.0 -o floor
rem cargo run --bin bake --release -- --scene "assets/meshes3/walkman/scene.gltf" --scale 10.0 -o walkman