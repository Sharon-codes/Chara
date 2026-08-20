# ==============================================================================
# 03_render_video.tcl - VMD Automated Frame Rendering & MP4 Assembly
# Renders QuickSurf MARTINI trajectory frames and stitches them using ffmpeg
# ==============================================================================

# 1. Load Topology and Trajectory
set gro_file "box.gro"
set xtc_file "production.xtc"

if {![file exists $gro_file]} {
    puts "Error: Topology file $gro_file not found."
    exit 1
}

if {![file exists $xtc_file]} {
    puts "Error: Trajectory file $xtc_file not found."
    exit 1
}

mol new $gro_file type gro waitfor all
mol addfile $xtc_file type xtc waitfor all

set molid [molinfo top get id]
set num_frames [molinfo top get numframes]
puts "Successfully loaded molecule $molid with $num_frames frames."

# 2. Configure Representation: QuickSurf & Color by Secondary Structure
mol delrep 0 $molid
mol representation QuickSurf 1.0 0.5 1.0 1.0
mol color Structure
mol selection "all"
mol material Glossy
mol addrep $molid

# Apply trajectory smoothing window (5 frames) to eliminate coarse-grained bead jitter
mol drawframes $molid 0 {smooth 5}

# Display Setup
display projection Orthographic
display depthcue off
axes location off
color Display Background white

# 3. Create Frames Directory and Render Snapshots
file mkdir "vmd_frames"

for {set i 0} {$i < $num_frames} {incr i} {
    animate goto $i
    display update
    set filename [format "vmd_frames/frame_%05d.tga" $i]
    render snapshot $filename
}

puts "Frame rendering complete. Stitching video with ffmpeg..."

# 4. Execute system call to ffmpeg for MP4 assembly at 30 fps
set output_mp4 "trajectory_6000ns.mp4"
set ffmpeg_cmd "ffmpeg -y -framerate 30 -i vmd_frames/frame_%05d.tga -c:v libx264 -pix_fmt yuv420p -crf 18 $output_mp4"

if {[catch {exec {*}$ffmpeg_cmd} msg]} {
    puts "ffmpeg execution completed: $msg"
} else {
    puts "Successfully rendered trajectory video to $output_mp4"
}

quit
