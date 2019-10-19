# cool-maps-with-rust

CRATES USED
- osmpbfreader (to read map data into Rust)
- nannou (for rendering graphics)
- petgraph for graphs (actually their djikstra/bellman-ford algorithm is broken, so had to re-implement myself)

PROBLEMS / ISSUES
- petgraph builds graphs correctly, but it's pathfinding algorithms (djikstra / bellman-ford) do not work correctly, so I had to write my own version
- nannou currently doesn't support framebuffers, so program runs very slowly once I zoom out to a wide view of the map (it tries to draw all the polygons every 1/60th of a second to the frame, rather than utilizing the fact that many of them don't change)

TO DO
- learn how to use vulkan directly, so I can implement my own framebuffer
- currently I can draw paths in real time from a point to the mouse cursor, but I want to build animated graph on the side of the visualization that shows path distance to various points (e.g. hospital, police station, etc.) in real time as you move the cursor (the bars in the bar graph would bounce up and down as you move the mouse)




