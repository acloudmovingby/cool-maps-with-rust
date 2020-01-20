use nannou::prelude::*;
use osmpbfreader::{Node, NodeId, Way};
use petgraph::graph::NodeIndex;
use petgraph::graph::*;
use petgraph::prelude::*;
use std::collections::hash_map::RandomState;
use std::collections::HashMap;
use map_project::read_map_data::read_buildings_from_map_data;
use map_project::read_map_data::road_graph_from_map_data;
use map_project::config::{Config,MapBounds,WindowDimensions};
use map_project::nannou_conversions::{convert_coord, batch_convert_coord};

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

fn setup_config() -> Config {
    let map_bounds = MapBounds {
        max_lon: -71.3919,
        min_lon: -71.4311,
        max_lat: 41.8300,
        min_lat: 41.8122,
    };
    let window_dimensions = calculate_window_dimensions(657.0,&map_bounds);
    let map_file_path = "/Users/christopherpoates/Downloads/rhode-island-latest.osm.pbf".to_string(); // RI
    //let map_file_path = "/Users/christopherpoates/Downloads/massachusetts-latest.osm.pbf".to_string(); // MA
    Config{map_bounds, window_dimensions, map_file_path}
}

fn calculate_window_dimensions(max_win_height: f32, map_bounds: &MapBounds) -> WindowDimensions {
    let lon_to_lat_ratio = ((map_bounds.max_lon-map_bounds.min_lon)/(map_bounds.max_lat-map_bounds.min_lat)) as f32;
    WindowDimensions{width: lon_to_lat_ratio*max_win_height, height: max_win_height}
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
    let config = setup_config();
    let _window = app
        .new_window()
        .with_dimensions(config.window_dimensions.width as u32, config.window_dimensions.height as u32)
        .view(view)
        .event(window_event)
        .build()
        .unwrap();

    let road_graph = road_graph_from_map_data(&config);
    let road_lines: Vec<Line> = color_roads(&road_graph, &config);
    let buildings = read_buildings_from_map_data(&config);
    let buildings: Vec<Vec<Point2>> = batch_convert_coord(&buildings, &config);

    Model {
        _window,
        buildings,
        road_lines,
    }
}

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

fn convert_coord2(node: &Node) -> Point2 {
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
                .map(|node| convert_coord2(&node))
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
fn color_roads(road_graph: &Graph<Node, f32, Directed>, config: &Config) -> Vec<Line> {
    let mut road_lines = Vec::new();
    for edge in road_graph.raw_edges() {
        let source = road_graph
            .node_weight(edge.source())
            .map(|node| convert_coord(node,config));
        let target = road_graph
            .node_weight(edge.target())
            .map(|node| convert_coord(node,config));

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
