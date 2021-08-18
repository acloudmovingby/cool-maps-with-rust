# cool-maps-with-rust

## OVERVIEW
This project explores different ways of rendering map data taken from OpenStreetMap.org. If you're interested in exploring the code, note that most of the different visualizations are broken up into separate binaries. Note that the "main" binary is simply a place where I experiment so that code is poorly organized.

## SCREENSHOTS
![basic map with buildings/roads](https://github.com/acloudmovingby/cool-maps-with-rust/blob/master/Screenshots/Just%20buildings%20and%20streets%20(Providence%2C%20RI).png)
Just a basic drawing of the buildings/streets in downtown Providence, RI. Notice that the map data is incomplete and so many buildings are missing :(

![path distance 2 (Providence,RI)](https://github.com/acloudmovingby/cool-maps-with-rust/blob/master/Screenshots/Path%20Distance%202%20(Providence%2C%20RI).png)
Min path distance from downtown Providence, RI. The waves of color don't represent any specific distance (like a quarter mile), although that would be trivial to implement. Because I was constantly trying the visualization at different scales, I just programmed it so the full hue change spans the whole width of the window, and saturation cycles ~10 times, making the appearance of "waves" propagating out.

![path distance 1 (Jamestown, RI)](https://github.com/acloudmovingby/cool-maps-with-rust/blob/master/Screenshots/Path%20distance%201%20(Jamestown%2C%20RI).png)

Min path distance from the island of Jamestown, RI. Here you can see path distance but without the "waves" (didn't cycle the saturation). Notice how the hues fan out radially from the tips of the bridges coming off the island. 

![building area (Boston, MA)](https://github.com/acloudmovingby/cool-maps-with-rust/blob/master/Screenshots/Shading%20buildings%20by%20area%20(Boston%2C%20MA).png)
This is downtown Boston / Cambridge. I colored the buildings based on their area. The largest buildings are blue, smallest are red. This one can be fun to play with.

![](https://github.com/acloudmovingby/cool-maps-with-rust/blob/master/Screenshots/Real-time%20pathfinding%20gif.gif)

Real time pathfinding (also min distance). Becomes too slow when you zoom out because no framebuffer (see problems section).

## ABOUT
CRATES USED
- osmpbfreader (to read map data into Rust)
- nannou (for rendering graphics)
- petgraph for graphs (actually their djikstra/bellman-ford algorithm is broken, so had to re-implement myself)

PROBLEMS / ISSUES
- map data, especially about buildings, wasn't as complete as I had anticipated
- petgraph builds graphs correctly, but it's pathfinding algorithms (djikstra / bellman-ford) either had some bugs or I was misusing them, so I had to write my own versions
- according to the people who answered me on slack, nannou currently doesn't support framebuffers, so program runs very slowly once I zoom out to a wide view of the map (it tries to redraw all the polygons every 1/60th of a second to the frame, rather than utilizing the fact that many of them don't change)

TO DO
- learn how to use vulkan directly, so I can implement my own framebuffer?
- currently I can draw paths in real time from a point to the mouse cursor, but I want to build animated graph on the side of the visualization that shows path distance to various points (e.g. hospital, police station, etc.) in real time as you move the cursor (the bars in the bar graph would bounce up and down as you move the mouse, while different color paths would show path to those buildings)




