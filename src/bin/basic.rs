use nannou::prelude::*;
use osmpbfreader::{Node, NodeId, Way};
use petgraph::graph::NodeIndex;
use petgraph::graph::*;
use petgraph::prelude::*;
use std::collections::hash_map::RandomState;
use std::collections::HashMap;

// AROUND MY HOUSE
/*
const MAX_LON: f64 = -71.10030;
const MIN_LON: f64 = -71.11010;
const MAX_LAT: f64 = 42.39705;
const MIN_LAT: f64 = 42.39263;*/

// DOWNTOWN BOSTON + CAMBRIDGE
/*
const MAX_LON: f64 = -71.0479;
const MIN_LON: f64 = -71.0871;
const MAX_LAT: f64 = 42.3704;
const MIN_LAT: f64 = 42.3527;*/

// NAHANT
/*
const MAX_LON: f64 = -70.8440;
const MIN_LON: f64 = -71.0009;
const MAX_LAT: f64 = 42.4808;
const MIN_LAT: f64 = 42.4070;*/

// NEWPORT / JAMESTOWN, RI
/*
const MAX_LON: f64 = -71.1856;
const MIN_LON: f64 = -71.4994;
const MAX_LAT: f64 = 41.5756;
const MIN_LAT: f64 = 41.4257;*/

// Near NEWPORT RI
/*
const MAX_LON: f64 = -71.3048;
const MIN_LON: f64 = -71.4133;
const MAX_LAT: f64 = 41.5108;
const MIN_LAT: f64 = 41.4749;*/

// PROVIDENCE DOWNTOWN
const MAX_LON: f64 = -71.3919;
const MIN_LON: f64 = -71.4311;
const MAX_LAT: f64 = 41.8300;
const MIN_LAT: f64 = 41.8122;

// LARGER PROVIDENCE AREA
/*
const MAX_LON: f64 = -71.2202;
const MIN_LON: f64 = -71.5340;
const MAX_LAT: f64 =  41.8831;
const MIN_LAT: f64 = 41.7403;*/

/*
const MAX_LON: f64 = -71.36522;
const MIN_LON: f64 = -71.38602;
const MAX_LAT: f64 = 41.53705;
const MIN_LAT: f64 = 41.52632;*/

//SMALL AREA FOR TESTING OF PETGRAPH BUG:
/*
const MAX_LON: f64 = -71.36531;
const MIN_LON: f64 = -71.37021;
const MAX_LAT: f64 = 41.53379;
const MIN_LAT: f64 = 41.53155;*/

/* Brown University
const MAX_LON: f64 = -71.3909;
const MIN_LON: f64 = -71.4105;
const MAX_LAT: f64 = 41.8282;
const MIN_LAT: f64 = 41.8192;*/

// Brown Frat squad
/*
const MAX_LON: f64 = -71.39901;
const MIN_LON: f64 = -71.40392;
const MAX_LAT: f64 = 41.82546;
const MIN_LAT: f64 = 41.82323;*/

const LON_RANGE: f64 = MAX_LON - MIN_LON;
const LAT_RANGE: f64 = MAX_LAT - MIN_LAT;

const MAX_WIN_WIDTH: f32 = 1357.0 * 0.7; // based on my laptop screen
const MAX_WIN_HEIGHT: f32 = 657.0 * 0.7; // based on my laptop screen

const WIN_W: f32 = ((MAX_WIN_HEIGHT as f64) / LAT_RANGE * LON_RANGE) as f32;
const WIN_H: f32 = MAX_WIN_HEIGHT;

fn main() {
    nannou::app(model).run();
}

struct Model {
    _window: window::Id,
    buildings: Vec<Vec<Point2>>, // each Vec represents a closed path of points describing the perimeter of the building
    road_lines: Vec<Line>,       // Line stores start, end, hue, saturation, alpha, thickness
}

/**
Sets up the initial state of the program before the nannou UI loop begins.
*/
fn model(app: &App) -> Model {
    let _window = app
        .new_window()
        .with_dimensions(WIN_W as u32, WIN_H as u32)
        .view(view)
        .event(window_event)
        .build()
        .unwrap();

    let filename = "/Users/christopherpoates/Downloads/rhode-island-latest.osm.pbf"; // RI
                                                                                     //let filename = "/Users/christopherpoates/Downloads/massachusetts-latest.osm.pbf"; // MA

    let r = std::fs::File::open(&std::path::Path::new(filename)).unwrap();
    let mut pbf = osmpbfreader::OsmPbfReader::new(r);

    let mut nodes = HashMap::new();
    let mut building_node_ids: Vec<Vec<NodeId>> = Vec::new();
    let mut all_roads: Vec<Way> = Vec::new();

    // READING MAP DATA
    for obj in pbf.par_iter().map(Result::unwrap) {
        match obj {
            osmpbfreader::OsmObj::Node(node) => {
                if is_in_outer_bounds(&node) {
                    nodes.insert(node.id, node);
                }
            }
            osmpbfreader::OsmObj::Way(way) => {
                if way.tags.contains_key("building") {
                    building_node_ids.push(way.nodes);
                } else if way.tags.contains_key("highway") {
                    all_roads.push(way);
                }
            }
            osmpbfreader::OsmObj::Relation(_rel) => {}
        }
    }

    let building_paths: Vec<Vec<Point2>> = node_ids_to_pts(building_node_ids, &nodes);
    //let building_paths: Vec<Vec<Point2>> = level_of_detail(1.0, building_paths); // if you want to make buildings simpler so it renders faster

    let buildings = building_paths;
    // now we take all_roads and remove all Ways that are not in map bounds to make a new collection: roads
    // roads thus represents all the Ways that have at least one node in the map bounds
    let mut roads: Vec<Way> = Vec::new();
    for road in all_roads {
        let any_in_bounds = road.nodes.iter().any(|node_id| {
            if nodes.contains_key(node_id) {
                let node = nodes.get(node_id).unwrap();
                is_in_bounds(node)
            } else {
                false
            }
        });

        // if any of the nodes in the road are in bounds, then keep the road for the graph
        if any_in_bounds {
            roads.push(road);
        }
    }

    // BUILD GRAPH
    // make hashmap of node ids to node indices
    // before adding a node, check to see if it exists
    // when you add an edge, use node index from hashmap
    let mut graph_node_indices = HashMap::new();
    let mut road_graph = Graph::<Node, f32, Directed>::new();
    for road in &roads {
        let mut prior_node_index = NodeIndex::new(0);
        for (i, node_id) in road.nodes.iter().enumerate() {
            // look up node using id
            if let Some(node) = nodes.get(node_id) {
                let cur_node_index: NodeIndex;
                if graph_node_indices.contains_key(node) {
                    cur_node_index = *graph_node_indices.get(node).unwrap();
                } else {
                    cur_node_index = road_graph.add_node(node.clone());
                    graph_node_indices.insert(node, cur_node_index);
                }

                // if it's not the first one, form an edge
                if i != 0 {
                    // find distances between the two points
                    let prior_node = road_graph
                        .node_weight(prior_node_index)
                        .expect("prior node should exist because we already traversed it");
                    let start_point = pt2(prior_node.lon() as f32, prior_node.lat() as f32);
                    let end_point = pt2(node.lon() as f32, node.lat() as f32);

                    road_graph.add_edge(
                        prior_node_index,
                        cur_node_index,
                        dist(start_point, end_point),
                    );
                    road_graph.add_edge(
                        cur_node_index,
                        prior_node_index,
                        dist(start_point, end_point),
                    );
                }
                prior_node_index = cur_node_index;
            }
        }
    }
    let road_lines: Vec<Line> = color_roads(&road_graph);

    Model {
        _window,
        buildings,
        road_lines,
    }
}

/**
Just
*/
fn window_event(_app: &App, _model: &mut Model, event: WindowEvent) {
    match event {
        KeyPressed(_key) => {}
        KeyReleased(_key) => {}
        MouseMoved(_pos) => {}
        MousePressed(_button) => {}
        MouseReleased(_button) => {}
        MouseEntered => {}
        MouseExited => {}
        MouseWheel(_amount, _phase) => {}
        Moved(_pos) => {}
        Resized(_size) => {}
        Touch(_touch) => {}
        TouchPressure(_pressure) => {}
        HoveredFile(_path) => {}
        DroppedFile(_path) => {}
        HoveredFileCancelled => {}
        Focused => {}
        Unfocused => {}
        Closed => {}
    }
}

/**
Nannou runs function 60 times per second to produce a new frame.
*/
fn view(app: &App, model: &Model, frame: Frame) -> Frame {
    let _win = app.window_rect();
    let draw = app.draw();

    for building in &model.buildings {
        let mut points: Vec<Point2> = Vec::new();
        for node in building {
            points.push(node.clone());
        }
        draw.polygon().points(points).hsv(0.6, 0.7, 0.5);
    }

    for road_line in model.road_lines.iter() {
        draw.line()
            .points(road_line.start, road_line.end)
            .thickness(road_line.thickness) // at some point draw according to geographical size ?
            .hsva(road_line.hue, road_line.saturation, 1.0, road_line.alpha);
    }

    draw.background().hsv(0.85, 0.3, 0.65);
    // Write to the window frame.
    draw.to_frame(app, &frame).unwrap();
    // Return the drawn frame.
    frame
}

fn is_in_bounds(node: &Node) -> bool {
    (node.lon() < MAX_LON)
        & (node.lon() > MIN_LON)
        & (node.lat() < MAX_LAT)
        & (node.lat() > MIN_LAT)
}

fn is_in_outer_bounds(node: &Node) -> bool {
    let outer_max_lon = MAX_LON + LON_RANGE * 1.0;
    let outer_min_lon = MIN_LON - LON_RANGE * 1.0;
    let outer_max_lat = MAX_LAT + LAT_RANGE * 1.0;
    let outer_min_lat = MIN_LAT - LAT_RANGE * 1.0;

    (node.lon() < outer_max_lon)
        & (node.lon() > outer_min_lon)
        & (node.lat() < outer_max_lat)
        & (node.lat() > outer_min_lat)
}

fn dist(pt1: Point2, pt2: Point2) -> f32 {
    ((pt1.x - pt2.x).powi(2) + (pt1.y - pt2.y).powi(2)).sqrt()
}

fn convert_coord(node: &Node) -> Point2 {
    let x = map_range(node.lon(), MIN_LON, MAX_LON, -WIN_W * 0.5, WIN_W * 0.5);
    let y = map_range(node.lat(), MIN_LAT, MAX_LAT, -WIN_W * 0.5, WIN_H * 0.5);
    pt2(x, y)
}

fn node_ids_to_pts(
    node_ids: Vec<Vec<NodeId>>,
    nodes: &HashMap<NodeId, Node, RandomState>,
) -> Vec<Vec<Point2>> {
    let mut building_paths = Vec::new();
    for building_path in node_ids {
        // if all the nodes in the building path are within bounds, then add the nodes to our list of buildings
        if building_path.iter().all(|x| nodes.contains_key(x)) {
            let building_path_as_points: Vec<Point2> = building_path
                .into_iter()
                .map(|x| nodes.get(&x).unwrap())
                .cloned()
                .map(|node| convert_coord(&node))
                .collect();
            building_paths.push(building_path_as_points);
        }
    }
    building_paths
}

/**
Currently not used (commented out above). min_size is the minimum pixel width a building has to span in order to be drawn.
This function returns the list of building points with any buildings smaller than the minimum removed.
*/
fn level_of_detail(min_size: f32, points: Vec<Vec<Point2>>) -> Vec<Vec<Point2>> {
    points
        .into_iter()
        .filter(|pts| {
            let mut min_y = std::f32::MAX;
            let mut max_y = -std::f32::MAX;
            let mut min_x = std::f32::MAX;
            let mut max_x = -std::f32::MAX;
            for pt in pts {
                if pt.x > max_x {
                    max_x = pt.x;
                }
                if pt.x < min_x {
                    min_x = pt.x;
                }
                if pt.y > max_y {
                    max_y = pt.y;
                }
                if pt.y < min_y {
                    min_y = pt.y;
                }
            }
            (max_x - min_x > min_size) && (max_y - min_y > min_size)
        })
        .collect()
}

/**
In this binary, this function draws the roads as white.
*/
fn color_roads(road_graph: &Graph<Node, f32, Directed>) -> Vec<Line> {
    let mut road_lines = Vec::new();
    for edge in road_graph.raw_edges() {
        let source = road_graph
            .node_weight(edge.source())
            .map(|node| convert_coord(node));
        let target = road_graph
            .node_weight(edge.target())
            .map(|node| convert_coord(node));

        if source.is_some() && target.is_some() {
            road_lines.push(Line {
                start: source.unwrap(),
                end: target.unwrap(),
                thickness: 2.0,
                hue: 1.0,
                saturation: 0.0,
                alpha: 1.0,
            });
        }
    }
    road_lines
}

/**
Stores the data I want to use when I draw a line using nannou's draw.line() builder, namely the end points and values for the color/thickness
*/
struct Line {
    start: Point2<f32>,
    end: Point2<f32>,
    thickness: f32,
    hue: f32,
    saturation: f32,
    alpha: f32,
}
