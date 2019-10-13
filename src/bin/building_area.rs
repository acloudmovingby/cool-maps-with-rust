use nannou::prelude::*;
use ordered_float::OrderedFloat;
use osmpbfreader::{Node, NodeId};
use std::collections::hash_map::RandomState;
use std::collections::HashMap;
use std::time::Instant;

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

// Brown Fraternity Quad
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

const HUE_MAX: f32 = 0.55; // the nannou API for hsv colors takes an f32 between 0.0 and 1.0
const HUE_MIN: f32 = 0.0;

fn main() {
    nannou::app(model).update(update).run();
}

struct Model {
    _window: window::Id,
    buildings: Vec<(Vec<Point2>, f32)>, // each Vec represents a closed path of points describing the perimeter of the building
}

fn model(app: &App) -> Model {
    let _window = app
        .new_window()
        .with_dimensions(WIN_W as u32, WIN_H as u32)
        .view(view)
        .event(window_event)
        .build()
        .unwrap();
    let t1 = Instant::now();

    let filename = "/Users/christopherpoates/Downloads/rhode-island-latest.osm.pbf"; // RI
                                                                                     //let filename = "/Users/christopherpoates/Downloads/massachusetts-latest.osm.pbf"; // MA

    let r = std::fs::File::open(&std::path::Path::new(filename)).unwrap();
    let mut pbf = osmpbfreader::OsmPbfReader::new(r);

    let mut nodes = HashMap::new();
    let mut building_node_ids: Vec<Vec<NodeId>> = Vec::new();

    println!("t1 before reading map data {}", t1.elapsed().as_secs());

    // READING MAP DATA
    for obj in pbf.par_iter().map(Result::unwrap) {
        match obj {
            osmpbfreader::OsmObj::Node(node) => {
                if is_in_bounds(&node) {
                    nodes.insert(node.id, node);
                }
            }
            osmpbfreader::OsmObj::Way(way) => {
                if way.tags.contains_key("building") {
                    building_node_ids.push(way.nodes);
                }
            }
            osmpbfreader::OsmObj::Relation(_rel) => {}
        }
    }
    let buildings: Vec<Vec<Point2>> = node_ids_to_pts(building_node_ids, &nodes);
    //let buildings: Vec<Vec<Point2>> = level_of_detail(1.0, buildings);

    println!("t1 after making the buildings: {}", t1.elapsed().as_secs());

    // create tuple pairs of buildings and their areas
    let mut buildings: Vec<(Vec<Point2>, f32)> = buildings
        .into_iter()
        .map(|points| {
            let area = polygon_area(&points).min(500.0);
            (points, area)
        })
        .collect();

    let min_area = buildings
        .iter()
        .map(|(_pts, area)| OrderedFloat(*area))
        .min()
        .unwrap()
        .0;
    let max_area = buildings
        .iter()
        .map(|(_pts, area)| OrderedFloat(*area))
        .max()
        .unwrap()
        .0;
    println!("max area is {}, and min area is {}", max_area, min_area);

    // map their areas to a hue (an f32 between 0.0 and 1.0)
    buildings = buildings
        .into_iter()
        .map(|(pts, area)| {
            let hue = map_range(area, min_area, max_area, HUE_MIN, HUE_MAX);
            (pts, hue)
        })
        .collect();

    Model { _window, buildings }
}

fn update(_app: &App, _model: &mut Model, _update: Update) {}

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

fn view(app: &App, model: &Model, frame: Frame) -> Frame {
    // Prepare to draw.
    let draw = app.draw();

    for building in &model.buildings {
        let mut points: Vec<Point2> = Vec::new();
        for node in building.0.iter() {
            points.push(node.clone());
        }
        draw.polygon().points(points).hsv(building.1, 1.0, 0.5);
    }

    draw.background().hsv(0.73, 0.0, 1.0);
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

// for performance reasons you can use this to remove buildings below a certain pixel size (both the width/height of the building must exceed the min_size)
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

fn polygon_area(points: &[Point2<f32>]) -> f32 {
    if points.is_empty() {
        0.0
    } else {
        let mut area = 0.0;
        let mut prev = points.iter().last().unwrap();

        for point in points {
            area += (point.x + prev.x) * (point.y - prev.y);
            prev = point;
        }
        (area / 2.0).abs()
    }
}
