use nannou::prelude::*;
use ordered_float::OrderedFloat;
use osmpbfreader::{Node, NodeId, Way};
use petgraph::algo::astar;
use petgraph::data::ElementIterator;
use petgraph::graph::NodeIndex;
use petgraph::graph::*;
use petgraph::prelude::*;
use petgraph::visit::NodeRef;
use rand::Rng;
use std::cmp::Ordering;
use std::collections::binary_heap::BinaryHeap;
use std::collections::hash_map::RandomState;
use std::collections::HashMap;
use std::f32::MAX;
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
/*
const MAX_LON: f64 = -71.3919;
const MIN_LON: f64 = -71.4311;
const MAX_LAT: f64 = 41.8300;
const MIN_LAT: f64 = 41.8122;*/

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
const MAX_LON: f64 = -71.39901;
const MIN_LON: f64 = -71.40392;
const MAX_LAT: f64 = 41.82546;
const MIN_LAT: f64 = 41.82323;


const LON_RANGE: f64 = MAX_LON - MIN_LON;
const LAT_RANGE: f64 = MAX_LAT - MIN_LAT;

const MAX_WIN_WIDTH: f32 = 1357.0 * 0.7; // based on my laptop screen
const MAX_WIN_HEIGHT: f32 = 657.0 * 0.7; // based on my laptop screen

const WIN_W: f32 = ((MAX_WIN_HEIGHT as f64) / LAT_RANGE * LON_RANGE) as f32;
const WIN_H: f32 = MAX_WIN_HEIGHT;

const HUE_MAX: f32 = 0.9; // the nannou API for hsv colors takes an f32 between 0.0 and 1.0
const HUE_MIN: f32 = 0.55;

fn main() {
    nannou::app(model).update(update).run();
}

struct Model {
    _window: window::Id,
    nodes: HashMap<NodeId, Node, RandomState>,
    buildings: Vec<Vec<Point2>>, // each Vec represents a closed path of points describing the perimeter of the building
    road_graph: Graph<Node, f32, Directed>,
    path_to_cursor: Vec<Line>,
    road_lines: Vec<Line>, // Line stores start, end, hue, saturation, alpha, thickness
    start: NodeIndex,      // the path goes from the mouse cursor to here
    target: Point2,
    closest_road_point: Point2,
    police_sts: Vec<Vec<Point2>>,
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
    let mut police_sts = Vec::new(); // police stations
    let mut all_roads: Vec<Way> = Vec::new();

    // TESTING
    let mut node_time = 0;
    let mut way_time = 0;
    let mut num_nodes = 0;
    let mut num_ways = 0;
    let mut num_way_nodes = 0;
    // TESTING

    println!("t1 before reading map data {}", t1.elapsed().as_secs());

    // READING MAP DATA
    for obj in pbf.par_iter().map(Result::unwrap) {
        match obj {
            osmpbfreader::OsmObj::Node(node) => {
                num_nodes += 1;

                let start = Instant::now();

                // this takes a long time, could possibly avoid by storing just the coordinates, not the full node object?
                //nodes.insert(node.id, node);
                if is_in_outer_bounds(&node) {
                    nodes.insert(node.id, node);
                }

                node_time += start.elapsed().as_micros();
            }
            osmpbfreader::OsmObj::Way(way) => {
                num_ways += 1;
                num_way_nodes += way.nodes.len();
                let start = Instant::now();
                if way.tags.contains("amenity", "police") {
                    police_sts.push(way.nodes.clone());
                }
                if way.tags.contains_key("building") {
                    building_node_ids.push(way.nodes);
                } else if way.tags.contains_key("highway") {
                    all_roads.push(way);
                }
                way_time += start.elapsed().as_micros();
            }
            osmpbfreader::OsmObj::Relation(rel) => {}
        }
    }

    println!(
        "The time spent on 'nodes' was {} seconds and on 'ways' was {} seconds.",
        node_time / 1_000_000,
        way_time / 1_000_000
    );
    println!(
        "The number of nodes was {}, the number of ways was {}, the number of way-nodes was {}.",
        num_nodes, num_ways, num_way_nodes
    );
    let nodes = nodes; // just reassign it so it's no longer mutable (is this good practice?)
    println!("police_sts len is {}", police_sts.len());
    println!("number of buildings is {}", building_node_ids.len());
    let building_paths: Vec<Vec<Point2>> = node_ids_to_pts(building_node_ids, &nodes);
    println!(
        "number of buildings after converting points is {}",
        building_paths.len()
    );
    //let building_paths: Vec<Vec<Point2>> = level_of_detail(1.0, building_paths);
    let police_sts = node_ids_to_pts(police_sts, &nodes);
    println!("police_sts len after converting it is {}", police_sts.len());

    println!("t1 after making the buildings: {}", t1.elapsed().as_secs());
    let _building_coordinates = building_paths;
    // now we take all_roads and remove all Ways that are not in map bounds to make a new collection: roads
    // roads thus represents all the Ways that have at least one node in the map bounds
    let mut roads: Vec<Way> = Vec::new();
    for road in all_roads {
        // TESTING
        for node_id in road.nodes.iter() {
            if nodes.contains_key(node_id) {
                let node = nodes.get(node_id).unwrap();
                if node.id.0 == 201212493 {
                    println!("#3 in roads Found left most, 201212493");
                }
                if node.id.0 == 201212498 {
                    println!("#3 in roads Found center, 201212498");
                }
                if node.id.0 == 201212502 {
                    println!("#3 in roads Found right, 201212502");
                }
            }
        }
        // END TESTING
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
    println!("t1 after making the roads: {}", t1.elapsed().as_secs());
    // testing
    let mut on_map = 0;
    let mut off_map = 0;
    // end testing

    // TESTING
    let mut target = pt2(0.0, 0.0);
    // END TESTING

    // BULD GRAPH
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
                // TESTING
                // see if node is in bounds, increment the proper value
                if is_in_bounds(&node) {
                    on_map += 1;
                } else {
                    off_map += 1;
                }
                // TESTING

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

                    // for any of those nodes, print the edge that is being added, so i can be sure it's adding those edges and no more
                    // for each node then, I should see two edges and those edge should be to the correct other points
                    // TESTING
                    let cur_node_id = node.id.0;
                    let other_node_id = prior_node.id.0;
                    if cur_node_id == 201212502 || other_node_id == 201212502 {
                        println!("EDGE: {}, {}", cur_node_id, other_node_id);
                    }

                    // END TESTING

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
    println!(
        "t1 after making the graph: {}. The graph has {} nodes and {} edges.",
        t1.elapsed().as_secs(),
        road_graph.node_count(),
        road_graph.edge_count()
    );

    println!(
        "{} road nodes are on the map and {} are off of it, making {} total",
        on_map,
        off_map,
        on_map + off_map
    );

    let node_indices: Vec<NodeIndex> = road_graph.node_indices().collect();
    let len = node_indices.len();
    println!("len = {}", len);
    let mut rng = rand::thread_rng();
    let rand1 = rng.gen_range(0, len);
    println!("rand1 = {}", rand1);
    let rand2 = rng.gen_range(0, len);
    println!("rand2 = {}", rand2);
    let start: NodeIndex = road_graph.node_indices().nth(rand1).unwrap();
    let end: NodeIndex = road_graph.node_indices().nth(rand2).unwrap();

    let path_to_cursor: Vec<Line> = Vec::new();

    // REDO GRAPH MIN DISTANCE STUFF
    let min_dist = djikstra_float(&road_graph, start);
    let max_weight = min_dist
        .values()
        .filter(|dist| dist.is_some())
        .map(|float_dist| (float_dist.unwrap() * 10000.) as u32)
        .max()
        .unwrap();

    let num_hue_cycles = 2; // number of times the hue range cycles through before reaching the farthest point.
    let sat_cycles = 10; // number of times saturation cycles through before reaching the farthest point

    let mut road_lines = Vec::<Line>::new();

    for edge in road_graph.raw_edges() {
        let source = road_graph
            .node_weight(edge.source())
            .map(|node| convert_coord2(node));
        let target = road_graph
            .node_weight(edge.target())
            .map(|node| convert_coord2(node));

        if source.is_some() && target.is_some() {
            // find weight (which is the f32 path distance from the "start" node to the source node of this edge)
            let weight = min_dist.get(&edge.target()).unwrap();
            // MAKES COLORS ROTATE CYCLICALLY
            let weight = (weight.unwrap_or(0.0) * 10000.0) as u32; // make an integer so you can use % operator
            let weight_hue = weight % (max_weight / num_hue_cycles);
            // TRANSFORM WEIGHT INTO HUE (0.0-1.0) / THICKNESS (1.0+) VALUES
            let hue = map_range(weight_hue, 0, max_weight / num_hue_cycles, HUE_MIN, HUE_MAX);

            // MAKE SATURATION ROTATE CYCLICALLY
            let weight = weight % (max_weight / sat_cycles); // make saturation cycle faster than hue
            let saturation = map_range(weight, 0, max_weight / sat_cycles, 0.5, 1.0);
            //let saturation = 1.0; // if you want no change/cycling of saturation

            // if you want thickness to vary by path distance from start:
            //let thickness = map_range(weight, 0, max_weight, 3.5, 1.0);
            let thickness = 1.0;
            let alpha = 1.0;

            road_lines.push(Line {
                start: source.unwrap(),
                end: target.unwrap(),
                thickness,
                hue,
                saturation,
                alpha,
            });
        }
    }
    /*
    // MIN DIST FIRST VERSION (WITH BUGGY BELLMAN_FORD ALGORITHM FROM PETGRAPH)
    // shortest path to all nodes in road graph from start node
    let min_paths = petgraph::algo::bellman_ford(&road_graph, start);
    let min_paths = min_paths.unwrap();

    let mut weight_iter = min_paths.0.iter();
    let mut node_iter = min_paths.1.iter();

    let max_weight = min_paths
        .0
        .iter()
        .filter(|x| x.is_finite())
        .map(|x| (x * 10000.0) as u32)
        .max()
        .unwrap();
    let num_hue_cycles = 2; // number of times the hue range cycles through before reaching the farthest point.
    let sat_cycles = 10; // number of times saturation cycles through before reaching the farthest point

    let min_paths: HashMap<Option<NodeIndex>, f32> = min_paths
        .1
        .iter()
        .copied()
        .zip(min_paths.0.iter().copied())
        .collect();

    let mut road_lines = Vec::<Line>::new();

    // FOR TESTING
    let mut in_bounds = 0;
    let mut partially_in_bounds = 0;
    let mut not_in_bounds = 0;

    for edge in road_graph.raw_edges() {
        let mut source_in_bounds = false; // TESTING
        let mut target_in_bounds = false; // TESTING
        let source = road_graph.node_weight(edge.source()).map(|node| {
            if is_in_bounds(&node) {
                source_in_bounds = true; // FOR TESTING
            };
            convert_coord2(node)
        }); // IMPORTANT
        let target = road_graph.node_weight(edge.target()).map(|node| {
            if is_in_bounds(&node) {
                target_in_bounds = true; // FOR TESTING
            };
            convert_coord2(node)
        }); // IMPORTANT

        // TESTING
        if source_in_bounds & target_in_bounds {
            in_bounds += 1;
        } else if source_in_bounds {
            partially_in_bounds += 1;
        } else {
            not_in_bounds += 1;
        }

        // IMPORTANT
        if source.is_some() && target.is_some() {
            // find weight (which is the f32 path distance from the "start" node to the source node of this edge)
            let weight_opt = min_paths.get(&Some(edge.source()));
                                                                  // if there was no weight (i.e. it was not connected to the start node), then make that edge transparent
            let alpha = 1.0; // if weight_opt.is_none() { 0.0 } else { 1.0 };

            let weight = weight_opt.unwrap_or(&80.0); // the path distance up to
                                                      //println!("Weight: {}", weight);
                                                      // MAKES COLORS ROTATE CYCLICALLY
            let weight = (weight * 10000.0) as u32; // make an integer so you can do % operator
            let weight_hue = weight % (max_weight / num_hue_cycles);
            // TRANSFORM WEIGHT INTO HUE (0.0-1.0) / THICKNESS (1.0+) VALUES
            let hue = map_range(weight_hue, 0, max_weight / num_hue_cycles, HUE_MIN, HUE_MAX);

            // MAKE SATURATION ROTATE CYCLICALLY
            let weight = weight % (max_weight / sat_cycles); // make saturation cycle faster than hue
            let saturation = map_range(weight, 0, max_weight / sat_cycles, 0.5, 1.0);
            //let saturation = 1.0; // if you want no change/cycling of saturation

            // if you want thickness to vary by path distance from start:
            //let thickness = map_range(weight, 0, max_weight, 3.5, 1.0);
            let thickness = 1.0;

            road_lines.push(Line {
                start: source.unwrap(),
                end: target.unwrap(),
                thickness,
                hue,
                saturation,
                alpha,
            });
        }
    }

    let mut in_bounds = in_bounds as f32;
    let mut partially_in_bounds = partially_in_bounds as f32;
    let mut not_in_bounds = not_in_bounds as f32;
    let total = in_bounds + partially_in_bounds + not_in_bounds;

    println!(
        "{}% are in bounds, {}% are partially in bounds and {}% are out of bounds",
        (in_bounds / total * 100.0) as u32,
        (partially_in_bounds / total * 100.0) as u32,
        (not_in_bounds / total * 100.0) as u32
    );*/

    let closest_road_point = pt2(0.0, 0.0);

    Model {
        _window,
        nodes,
        buildings: _building_coordinates,
        road_graph,
        path_to_cursor,
        road_lines,
        start,
        target,
        closest_road_point,
        police_sts,
    }
}

fn update(_app: &App, model: &mut Model, _update: Update) {
    let mouse_pt = pt2(_app.mouse.x, _app.mouse.y);

    let node_closest_to_mouse = model.road_graph.node_indices().min_by_key(|ix| {
        let node = model.road_graph.node_weight(*ix).unwrap();
        let node_pt = convert_coord2(node);
        dist(mouse_pt, node_pt) as u32 // cast because you can't compare floats. these represent pixesl on the screen, so distance should be at least 1.0 and if it's less than that, that's fine
    });

    if node_closest_to_mouse.is_some() {
        let node_closest_to_mouse = node_closest_to_mouse.unwrap();

        model.closest_road_point =
            convert_coord2(&model.road_graph.node_weight(node_closest_to_mouse).unwrap());

        let path_as_node_indices = astar(
            &model.road_graph,
            model.start,
            |finish| finish == node_closest_to_mouse,
            |e| *e.weight(),
            |_| 0.0,
        );

        // build the Lines that form the edge from the current path
        if path_as_node_indices.is_some() {
            model.path_to_cursor = path_as_node_indices
                .unwrap()
                .1
                .iter()
                .enumerate()
                .fold(
                    (Vec::new(), pt2(0.0, 0.0)),
                    |(mut vec, prior_pt), (i, node_ix)| {
                        let end_pt =
                            convert_coord2(model.road_graph.node_weight(*node_ix).unwrap());
                        if i != 0 {
                            let line = Line {
                                start: prior_pt,
                                end: end_pt,
                                thickness: 5.0,
                                hue: 0.0,
                                saturation: 0.0,
                                alpha: 1.0,
                            };
                            vec.push(line);
                        }
                        (vec, end_pt)
                    },
                )
                .0;
        }
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

fn view(app: &App, model: &Model, frame: Frame) -> Frame {
    let win = app.window_rect();
    // Prepare to draw.
    let draw = app.draw();
    let t = app.time;


        for building in &model.buildings {
            let mut points: Vec<Point2> = Vec::new();

            for node in building {
                let x = node.clone();
                points.push(node.clone());
            }
            // just draws the first point as a small ellipse (with oscillating hue)
            let oscillator = map_range(t.cos(), -1.0, 1.0, 0.0, 1.0);
            let area = polygon_area(&points).min(150.0);

            let hue = map_range(area, 15.0,150.0, 0.0,0.6);
            draw.polygon()
                .points(points)
                .hsv(hue, 1.0, 0.5)

            /*.rotate(-t * 0.1)*/;
        }

    // DRAW POLICE STATIONS
    /*
    for police_station in model.police_sts.iter() {
        draw.polygon().points(police_station.clone()).color(WHITE);
    }*/

    // DRAW ROADS
    // range of hues: 0.55 (cyan) 0.9 (pink)

    for road_line in model.road_lines.iter() {
        // makes the hues move out in ripples
        let hue_max_int = (HUE_MAX * 1000.0) as u32; // should be a number between 0 and 100 (hues in the hsv are taken as f32s between 0.0 and 1.0)
        let hue_int = (road_line.hue * 1000.0) as u32;
        let hue = ((t * 40.0) as u32 + hue_int) % hue_max_int;
        let hue = map_range(hue, 0, 1000, HUE_MAX, HUE_MIN);

        // saturation done same as hue above, so they move in sync
        let sat_max = 1000;
        let sat = (road_line.saturation * 1000.0) as u32;
        let sat = ((t * 40.0) as u32 + sat) % sat_max;
        let sat = map_range(sat, 0, 1000, 1.0, 0.5);

        draw.line()
            .points(road_line.start, road_line.end)
            .thickness(road_line.thickness) // at some point draw according to geographical size ?
            .hsva(road_line.hue, road_line.saturation, 1.0, road_line.alpha);
    }

    // DRAW PATH TO CURSOR
    for line in model.path_to_cursor.iter() {
        draw.line()
            .points(line.start, line.end)
            .thickness(line.thickness)
            .hsva(line.hue, line.saturation, 1.0, line.alpha);
    }

    // DRAW START NODE
    /*
        let start = model.road_graph.node_weight(model.start);
        if start.is_some() {
            let start = convert_coord2(start.unwrap());
            draw.ellipse()
                .xy(start)
                .radius(10.0)
                .hsva(1.0, 1.0, 1.0, 0.4);
        }
    */
    /*

    // DRAW ROAD NODES AS ELLIPSES (ACTUALLY START POINT OF ROAD LINES)
    for line in model.road_lines.iter() {
        let pt = line.start;
        draw.ellipse().radius(1.0).x(pt.x).y(pt.y).color(BLUE);
    }*/

    // DRAW ROAD NODES FROM ROAD_GRAPH
    println!("num nodes: {}",model.road_graph.raw_nodes().len());
    model.road_graph.raw_nodes().iter().map(|node| {
        println!("hey");
        let pt = convert_coord2(&node.weight);
        draw.ellipse().x(pt.x).y(pt.y).color(GREEN).radius(0.25);
    });

    //SELECT ROAD NODE CLOSEST MOUSE CURSOR

    /*
    let mouse_pt = pt2(app.mouse.x, app.mouse.y);

    let closest_road_line = model
        .road_lines
        .iter()
        .min_by_key(|x| dist(x.start, mouse_pt) as u32)
        .unwrap();

    let closest_road_point =
        if dist(closest_road_line.start, mouse_pt) < dist(closest_road_line.end, mouse_pt) {
            closest_road_line.start
        } else {
            closest_road_line.end
        };*/

    // RED ELLIPSE AT CURSOR

    draw.ellipse()
        .x(model.closest_road_point.x)
        .y(model.closest_road_point.y)
        .radius(5.0)
        .color(RED);

    /* REAL-TIME PATH DRAWING: */
    /*
        let mouse_pt = pt2(app.mouse.x, app.mouse.y);
        let closest_road_node = model.road_graph.node_indices()
            .min_by_key(|&ix| {
                let node = model.road_graph.node_weight(ix).unwrap();
                draw
                let node_pt = convert_coord2(node);
                (dist(mouse_pt, node_pt)*1000.0) as u32
            });
    **/
    /*
    if closest_road_node.is_some() {
        let pt = convert_coord2(model.road_graph.node_weight(closest_road_node.unwrap()).unwrap());
        draw.ellipse()
            .x(pt.x)
            .y(pt.y)
            .radius(5.0)
            .color(RED);
    }*/

    // DRAW ELLIPSE AT CURSOR
    /*draw.ellipse()
            .x(app.mouse.x)
            .y(app.mouse.y)
            .radius(15.0)
            .rgba(1.0,1.0,1.0,0.5);
    */
    /*



    // DRAW PATH (CURRENTLY BETWEEN RANDOM ROAD NODES)
    let mut prior_node_index = NodeIndex::new(0);
    for (i, cur_node_index) in model.path.iter().enumerate() {
        match model.road_graph.node_weight(*cur_node_index) {
            Some(node) => {
                // COLORING
                //let color = BLACK;
                let weight = model.road_coloring.get(&Some(cur_node_index.clone()));
                /*let color = weight.map_or(BLACK, |w|
                    Alpha {
                    color: Rgb {
                        red: 1.0,
                        green: 200.0/255.0,
                        blue: (w*2.0)/255.0,
                    },
                    alpha: 1.0,
                });*/
                let weight = weight.unwrap_or(&0.0);

                let weight = map_range(*weight, 0.0, 80.0, 0.5,0.75);

                // DRAWING ROAD
                // if it's not the first one, draw an edge
                if is_in_bounds(node) {
                    if i != 0 {
                        draw.line()
                            .points(
                                convert_coord(
                                    model.road_graph.node_weight(prior_node_index).unwrap(),
                                    &win,
                                ),
                                convert_coord(node, &win),
                            )
                            .x(-win.w() * 0.5)
                            .y(-win.h() * 0.5)
                            .thickness(4.0)
                            .hsva(weight, 1.0,1.0,1.0)
                            /*.color(color)*/;
                    }
                }
                prior_node_index = *cur_node_index;
            }
            None => {}
        }
    }*/

    /*
    // DRAWS BOUNDING BOX AT MAX/MIN COORDINATES, DOESN"T WORK YET
    let ne = pt2(WIN_W*0.5, WIN_H*0.5);
    let nw = pt2(-WIN_W*0.5, WIN_H*0.5);
    let sw = pt2(-WIN_W*0.5, -WIN_H*0.5);
    let se = pt2(WIN_W*0.5, -WIN_H*0.5);
    draw.quad()
        .points(ne, nw, sw, se)
        .color(TRANSPARENT);
        */

    // DRAW TARGET NODE
    /*draw.ellipse()
        .x(model.target.x)
        .y(model.target.y)
        .radius(5.0)
        .color(YELLOW);*/
    // Clear the background to pink.

    
    draw.background().hsv(0.73, 0.55, 0.06);
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

fn level_of_detail(min_size: f32, points: Vec<Vec<Point2>>) -> Vec<Vec<Point2>> {
    let simplified_points = points
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
        .collect();
    simplified_points
}

// stores the data I want to use when I draw a line using nannou's draw.line() builder, namely the end points and values for the color/thickness
struct Line {
    start: Point2<f32>,
    end: Point2<f32>,
    thickness: f32,
    hue: f32,
    saturation: f32,
    alpha: f32,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
struct Dist_Float {
    dist: OrderedFloat<f32>,
    node: NodeIndex,
}

impl Ord for Dist_Float {
    fn cmp(&self, other: &Dist_Float) -> Ordering {
        other
            .dist
            .cmp(&self.dist)
            .then_with(|| self.node.index().cmp(&other.node.index()))
    }
}

impl PartialOrd for Dist_Float {
    fn partial_cmp(&self, other: &Dist_Float) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

fn djikstra_float(
    g: &Graph<Node, f32, Directed>,
    start: NodeIndex<u32>,
) -> HashMap<NodeIndex, Option<f32>, RandomState> {
    // check if the node exists in the graph (there is no contains(nodeindex) method in petgraph (why??), so this just retrieves the node's weight as an option and checks if that's None or not)
    if g.node_weight(start).is_some() {
        let mut min_dist: HashMap<NodeIndex, f32, RandomState>; // final return value; maps node indices to their min distance from the start
        let mut edge_dist: BinaryHeap<Dist_Float>; // potential edges to explore; each Dist stores a node and a distance leading up to that node (which may not be the min distance)
                                                   // Djikstra's formula works by always selecting the minimum of those potential edges (hence why it's a BinaryHeap (Priority Queue)

        // initialize min_dist
        min_dist = HashMap::new();
        for node in g.node_indices() {
            min_dist.insert(node.clone(), std::f32::INFINITY);
        }
        min_dist.insert(start, 0.0);

        // initialize edge_dist, the priority queue (binary heap), with all outgoing edges from start
        let edges_from_start = g.edges_directed(start, Outgoing);
        edge_dist = edges_from_start
            .map(|edge| Dist_Float {
                node: edge.target(),
                dist: OrderedFloat(*edge.weight()),
            })
            .collect();

        // traverse graph, adding one node at a time (choose the one with the lowest accumulated path distance)
        while !edge_dist.is_empty() {
            let next = edge_dist.pop().unwrap(); // take minimum

            // if it's lower than the currently saved min distance value, then update min_dist and add its edges to the edge priority queue
            if next.dist.into_inner() < *min_dist.get(&next.node).unwrap() {
                min_dist.insert(next.node, next.dist.into_inner());

                let outgoing = g.edges_directed(next.node, Outgoing);
                let outgoing: Vec<Dist_Float> = outgoing
                    .map(|edge| Dist_Float {
                        node: edge.target(),
                        dist: OrderedFloat(*edge.weight() + next.dist.into_inner()),
                    })
                    .collect();
                for dist in outgoing {
                    edge_dist.push(dist);
                }
            }
        }

        let min_dist: HashMap<NodeIndex, Option<f32>, RandomState> = min_dist
            .into_iter()
            .map(|(node, dist)| {
                if dist == std::f32::INFINITY {
                    (node, None)
                } else {
                    (node, Some(dist))
                }
            })
            .collect();

        min_dist
    } else {
        HashMap::new()
    }
}

fn polygon_area(points: &Vec<Point2<f32>>) -> f32 {
    if points.len() == 0 {
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
